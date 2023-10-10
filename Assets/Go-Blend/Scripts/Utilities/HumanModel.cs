using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;

namespace Go_Blend.Scripts.Utilities
{
    public static class HumanModel
    {
        private static Dictionary<int, float[]> _previousLookUps;
        private static List<float[]> _dataset;
        private static List<float> _datasetArousalValues;
        public static List<float> Minimums, Maximums;
        private static string _clusterID;
        private static int _k;
        
        public static void Init(string cluster, int k)
        {
            _clusterID = cluster;
            _k = k;
            LoadDict();
            _dataset = new List<float[]>();
            _datasetArousalValues = new List<float>();
            using var reader = new StreamReader($"{Application.persistentDataPath}/Humans/Solid_MinMax_250ms_{cluster}.csv");
            using var session = new StreamReader($"{Application.persistentDataPath}/Humans/Solid_SessionNorm_250ms_{cluster}.csv");
            reader.ReadLine(); //Ignore headers
            session.ReadLine(); //Ignore headers
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = line.Split(',');
                var entry = new float[32];
                var counter = 0;
                for (var i = 0; i < values.Length; i++)
                {
                    if (i > 0 && i < 33)
                        entry[counter++] = float.Parse(values[i]);
                }
                _dataset.Add(entry);
            }
            while (!session.EndOfStream)
            {
                var sessionLine = session.ReadLine();
                var sessionValues = sessionLine.Split(',');
                _datasetArousalValues.Add(float.Parse(sessionValues[33]));
            }
            GetMinMax(cluster);
        }
        
        private static void LoadDict()
        {
            try
            {
                while (true)
                {
                    try
                    {
                        var bf = new BinaryFormatter();
                        var file = File.Open($"{Application.persistentDataPath}/Humans/Model_Dictionary_{_clusterID}_K{_k}.file", FileMode.Open);
                        _previousLookUps = (Dictionary<int, float[]>)bf.Deserialize(file);
                        //Debug.Log("Dictionary loaded!");
                        file.Close();
                        break;
                    } catch (AccessViolationException){Debug.LogError("Can't access database, trying again...");}

                }

            }
            catch (FileNotFoundException)
            {
                //Debug.Log("No existing dictionary found. Creating a new one.");
                _previousLookUps = new Dictionary<int, float[]>();
            }
        }

        public static void SaveDict()
        {
            var bf = new BinaryFormatter();
            var file = File.Create($"{Application.persistentDataPath}/Humans/Model_Dictionary_{_clusterID}_K{_k}.file");
            bf.Serialize(file, _previousLookUps);
            file.Close(); 
            LoadDict();
        }
        
        private static void GetMinMax(string add)
        {
            using var reader = new StreamReader($"{Application.persistentDataPath}/Humans/Solid_NoNorm_250ms_{add}.csv");
            reader.ReadLine();
            Minimums = new List<float>();
            Maximums = new List<float>();
            for (var i = 0; i < 32; i++)
            {
                Minimums.Add(Mathf.Infinity);
                Maximums.Add(Mathf.NegativeInfinity);
            }
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                var values = line.Split(',');

                for (var i = 0; i < values.Length; i++)
                {
                    if (i is <= 0 or >= 33) continue;
                    if (float.Parse(values[i]) > Maximums[i-1]) Maximums[i-1] = float.Parse(values[i]);
                    else if (float.Parse(values[i]) < Minimums[i-1]) Minimums[i-1] = float.Parse(values[i]);
                }
            }
        }

        public static float[] GetClosestHuman(float [] surrogateVector)
        {
            var stateHash = Utility.GetHashFromArray(surrogateVector);
            if (_previousLookUps.ContainsKey(stateHash))
            {
                //Debug.Log("Returning existing value...");
                return _previousLookUps[stateHash];
            }
            //Debug.Log("No existing query found, generating new one...");
            var subset = _datasetArousalValues.Count;
            var distances = new Dictionary<int, double>();
            for (var entry = 0; entry < subset; entry++)
            {
                distances.Add(entry, 1 / Utility.Distance(surrogateVector, _dataset[entry]));
            }
            var sortedValues = from entry in distances orderby entry.Value descending select entry;
            var counter = 0;
            var sum = 0.0;
            foreach (var value in sortedValues)
            {
                sum += value.Value;
                if (++counter == _k)
                    break;
            }
            counter = 0;
            var arousalValue = 0.0;
            var arousalSample = new List<float>();
            foreach (var value in sortedValues)
            {
                arousalSample.Add(_datasetArousalValues[value.Key]);
                arousalValue += _datasetArousalValues[value.Key] * (value.Value / sum);
                if (++counter == _k) break;
            }

            var humanArousal = new[] { Utility.Deviation(arousalSample.ToArray()), (float)arousalValue };
            _previousLookUps.Add(stateHash, humanArousal);
            return humanArousal;
        }
        
    }
}
