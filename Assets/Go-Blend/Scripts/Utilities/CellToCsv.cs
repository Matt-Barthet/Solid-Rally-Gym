using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using UnityEngine;

namespace Go_Blend.Scripts.Utilities
{
    public class CellToCsv : MonoBehaviour
    {

        public string[] clusters = { "Cluster0", "Cluster1", "Cluster2", "Cluster3" };
        // public string[] clusters = { "Maximize Arousal", "Minimize Arousal" };
        // public string[] rewardFunctions = { "R0.0", "R0.5", "R1.0" };
        public string[] rewardFunctions = { "R0.0" };
        public int [] runs;

        private string counter;

        private void Start()
        {
            foreach (var cluster in clusters)
            {

                HumanModel.Init(counter, 5);

                foreach(var rewardFunction in rewardFunctions)
                {
                    foreach (var i in runs)
                    {
                        string path = $"./Evaluation/{cluster}/{rewardFunction}/Run{i}/Data/Cells";
                        var info = new DirectoryInfo(path);
                        var fileInfo = info.GetFiles();

                        foreach (var file in fileInfo)
                        {
                            var split = file.ToString().Split('/');
                            var filename = split[split.Length - 1];

                            if (!filename.Contains(".csv"))
                            {
                                BinaryFormatter bf = new();
                                FileStream cellFile = File.Open($"{path}/" + filename, FileMode.Open);

                                try
                                {
                                    object cell = bf.Deserialize(cellFile);
                                    var save = (StateSaveLoad.Save)cell;
                                    var key = filename.Split(".")[0];
                                    GoExplore.SaveCell(save.cell, $"{path}/{key}");
                                }
                                catch (Exception e)
                                {
                                    continue;
                                }

                            
                                /*TextWriter tw = new StreamWriter($"{path}/{key}_feat.csv");
                            for (var j = 0; j < save.cell.surrogateTrace.Count; j++)
                            {
                                string line = "";
                                for (var k = 0; k < 31; k++)
                                {
                                    line += $"{save.cell.surrogateTrace[j][k]},";
                                }
                                line += $"{save.cell.surrogateTrace[j][31]}\n";
                                tw.Write(line);
                            }
                            tw.Close();  */
                            }
                        }
                    }
                }
            }
        }

        public static Dictionary<int, Cell> LoadArchiveFromFile(string prefix="")
        {
            var cellArchive = new Dictionary<int, Cell>();
            var info = new DirectoryInfo($".{prefix}/Data/Cells");
            var fileInfo = info.GetFiles();
            foreach (var file in fileInfo)
            {
                var split = file.ToString().Split('/');
                var filename = split[split.Length - 1];
                if (!filename.Contains(".save")) continue;
                var bf = new BinaryFormatter();
                var cellFile = File.Open("./Data/Cells/" + filename, FileMode.Open);
                var save = (StateSaveLoad.Save)bf.Deserialize(cellFile);
                var cell = save.cell;
                var key = filename.Split(".")[0];
                cellArchive.Add(int.Parse(key), cell);
            }
            return cellArchive;
        }

    }
}
