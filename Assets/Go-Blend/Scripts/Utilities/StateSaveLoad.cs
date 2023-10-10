using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Go_Blend.Scripts.Car_Controllers;
using Go_Blend.Scripts.Car_Controllers.Base_Classes;
using MoreMountains.HighroadEngine;
using UnityEngine;

namespace Go_Blend.Scripts.Utilities
{
    public class StateSaveLoad: MonoBehaviour
    {
        public static GoExplore backEnd;
        private static GameObject _player;
        private static RaceManager raceManager;

        public static string CurrentPath;
    
        private void Start()
        {
            raceManager = GetComponent<RaceManager>();
        }

        public static void SetCurrentPath(string folder)
        {
            CurrentPath = folder;
        }
    
        public static void SaveEnvironment(string filename)
        {
            Save save = new(raceManager);
            BinaryFormatter bf = new BinaryFormatter();
            FileStream file = File.Create($"{CurrentPath}/Cells/{filename}.save");
            bf.Serialize(file, save);
            file.Close();
        }
    
        public static void Reset()
        {
            _player = GameObject.FindGameObjectWithTag("Player");
            RaceManager._currentGameTime = 0;
            foreach (var bot in raceManager.bots)
                bot.GetComponent<VehicleAI>().Initialize();
            _player.GetComponent<GoBlendReplayController>()?.Initialize(); 
            _player.GetComponent<GoBlendCarController>()?.Initialize();
            _player.GetComponent<GoBlendSolidController>()?.Initialize();
        }

        public static Cell LoadCell(string filename)
        {
            var bf = new BinaryFormatter();
            print($"{CurrentPath}/Cells/{filename}.save");
            var file = File.Open($"{CurrentPath}/Cells/{filename}.save", FileMode.Open);
            var save = (Save)bf.Deserialize(file);
            return save.cell;
        }
    
        public static void LoadEnvironment(string filename, bool ignoreCWD = false)
        {
            _player = GameObject.FindGameObjectWithTag("Player");
            backEnd = _player.GetComponent<GoBlendBaseController>()?.backEnd;

            var bf = new BinaryFormatter();
            Debug.Log($"{Application.persistentDataPath}/{filename}.save");
            var file = File.Open(ignoreCWD ? $"{Application.persistentDataPath}/{filename}.save" : $"{Application.persistentDataPath}/{CurrentPath}/{filename}.save", FileMode.Open);
            var save = (Save)bf.Deserialize(file);

            RaceManager._currentGameTime = save.currentTime;

            _player.GetComponent<BaseController>().Score = save.currentScore;
            _player.GetComponent<BaseController>()._currentWaypoint = save.currentWayPoint;
            _player.GetComponent<BaseController>()._lastWaypointCrossed = save.lastWayPoint;
            _player.GetComponent<BaseController>().CurrentLap = save.currentLap;
            _player.GetComponent<BaseController>().CurrentSteeringAmount = save.currentSteering;
            _player.GetComponent<BaseController>().CurrentGasPedalAmount = save.currentGas;

            print($"{save.PlayerProperties[0, 0]}, {save.PlayerProperties[0, 1]}, {save.PlayerProperties[0, 2]}");
            _player.transform.position = new Vector3(save.PlayerProperties[0, 0], save.PlayerProperties[0, 1], save.PlayerProperties[0, 2]);
            _player.transform.eulerAngles = new Vector3(save.PlayerProperties[1, 0], save.PlayerProperties[1, 1], save.PlayerProperties[1, 2]);
            _player.GetComponent<Rigidbody>().velocity = new Vector3(save.PlayerProperties[2, 0], save.PlayerProperties[2, 1], save.PlayerProperties[2, 2]);
            _player.GetComponent<Rigidbody>().angularVelocity = new Vector3(save.PlayerProperties[3, 0], save.PlayerProperties[3, 1], save.PlayerProperties[3, 2]);

            if (raceManager != null)
            {
                for (var i = 0; i < raceManager.bots.Count; i++)
                {
                    raceManager.bots[i].transform.position = new Vector3(save.BotProperties[i][0, 0],
                        save.BotProperties[i][0, 1], save.BotProperties[i][0, 2]);
                    raceManager.bots[i].transform.eulerAngles = new Vector3(save.BotProperties[i][1, 0],
                        save.BotProperties[i][1, 1], save.BotProperties[i][1, 2]);
                    raceManager.bots[i].GetComponent<Rigidbody>().velocity = new Vector3(save.BotProperties[i][2, 0],
                        save.BotProperties[i][2, 1], save.BotProperties[i][2, 2]);
                    raceManager.bots[i].GetComponent<Rigidbody>().angularVelocity = new Vector3(save.BotProperties[i][3, 0],
                        save.BotProperties[i][3, 1], save.BotProperties[i][3, 2]);

                    raceManager.bots[i].GetComponent<BaseController>().Score = save.botScores[i];
                    raceManager.bots[i].GetComponent<BaseController>().CurrentLap = save.botLaps[i];
                    raceManager.bots[i].GetComponent<BaseController>().CurrentGasPedalAmount = save.botCurrentGas[i];
                    raceManager.bots[i].GetComponent<BaseController>().CurrentSteeringAmount = save.botCurrentSteering[i];
                    raceManager.bots[i].GetComponent<BaseController>()._currentWaypoint = save.botCurrentWayPoints[i];
                    raceManager.bots[i].GetComponent<BaseController>()._lastWaypointCrossed = save.botLastWayPoints[i];
                    raceManager.bots[i].GetComponent<VehicleAI>()._currentWaypoint = save.botCurrentAIPoint[i];
                    raceManager.bots[i].GetComponent<VehicleAI>()._targetWaypoint = raceManager.bots[i].GetComponent<VehicleAI>()._AIWaypoints[save.botCurrentAIPoint[i]];

                    raceManager.bots[i].GetComponent<SolidController>().previousPos = new Vector3(save.botPrevious[i][0], save.botPrevious[i][1], save.botPrevious[i][2]);
                    raceManager.bots[i].GetComponent<SolidController>().previousEurler = new Vector3(save.botPrevEuler[i][0], save.botPrevEuler[i][1], save.botPrevEuler[i][2]);

                    for (int j = 0; j < 4; j++)
                        raceManager.bots[i].GetComponent<VehicleAI>().solidWheels[j].SetProperties(save.botWheels[i][j][0], save.botWheels[i][j][1], save.botWheels[i][j][2], save.botWheels[i][j][3]);
                }
            }

            var carController = _player.GetComponent<GoBlendSolidController>();
            carController.currentPoint = save.currentPoint;
            carController.bestScore = save.bestScore;
            carController.currentTime = 0.25f;

            _player.GetComponent<SolidController>().previousPos = new Vector3(save.previousPos[0], save.previousPos[1], save.previousPos[2]);
            _player.GetComponent<SolidController>().previousEurler = new Vector3(save.previousEuler[0], save.previousEuler[1], save.previousEuler[2]);

            carController.ExplorePoints.Clear();
            carController.Rotations.Clear();
            carController.Velocities.Clear();
            carController.Angulars.Clear();

            foreach (var point in save.explorePoints)
                carController.ExplorePoints.Add(new [] { point[0], point[1], point[2] });        
    
            foreach (var point in save.rotation)
                carController.Rotations.Add(new [] { point[0], point[1], point[2] });

            foreach (var point in save.velocities)
                carController.Velocities.Add(new [] { point[0], point[1], point[2] });

            foreach (var point in save.angulars)
                carController.Angulars.Add(new [] { point[0], point[1], point[2] });

            try
            {
                for (int i = 0; i < 4; i++)
                    carController.solidWheels[i].SetProperties(save.wheels[i][0], save.wheels[i][1], save.wheels[i][2],
                        save.wheels[i][3]);
            }
            catch (Exception)
            {
                Debug.LogError("Caught wheel error in loading script!");
            }

            file.Close();
        }

        [Serializable]
        public class Save
        {
            public float currentTime, bestScore;
            public int currentScore, currentLap, currentWayPoint, lastWayPoint, currentPoint;
            public float currentSteering, currentGas;

            public float[,] PlayerProperties;
            public List<float[,]> BotProperties;

            public List<int> botScores, botLaps, botCurrentWayPoints, botLastWayPoints, botCurrentAIPoint;
            public List<float> botCurrentSteering, botCurrentGas;

            public List<float[]> explorePoints, rotation, velocities, angulars;

            public float[] previousPos, previousEuler;
            public List<float[]> wheels;

            public List<float[]> botPrevious, botPrevEuler;
            public List<List<float[]>> botWheels;

            public Cell cell;

            public Save(RaceManager raceManager)
            {
                _player = GameObject.FindGameObjectWithTag("Player");
                backEnd = _player.GetComponent<GoBlendBaseController>()?.backEnd;
                currentTime = RaceManager._currentGameTime;
                currentScore = _player.GetComponent<BaseController>().Score;
                currentLap = _player.GetComponent<BaseController>().CurrentLap;
                currentWayPoint = _player.GetComponent<BaseController>()._currentWaypoint;

                var position = _player.transform.position;
                var eulerAngles = _player.transform.eulerAngles;
                var velocity = _player.GetComponent<Rigidbody>().velocity;
                var angularVelocity = _player.GetComponent<Rigidbody>().angularVelocity;

                PlayerProperties = new [,] { 
                    {position.x, position.y, position.z}, 
                    {eulerAngles.x, eulerAngles.y, eulerAngles.z}, 
                    {velocity.x, velocity.y, velocity.z},
                    {angularVelocity.x, angularVelocity.y, angularVelocity.z}
                };

                BotProperties = new List<float[,]>();
                botScores = new List<int>();
                botLaps = new List<int>();
                botCurrentGas = new List<float>();
                botCurrentSteering = new List<float>();
                botCurrentWayPoints = new List<int>();
                botLastWayPoints = new List<int>();
                botCurrentAIPoint = new List<int>();
                botWheels = new List<List<float[]>>();
                botPrevEuler = new List<float[]>();
                botPrevious = new List<float[]>();

                if (raceManager != null)
                {
                    foreach (var t in raceManager.bots)
                    {
                        var botPosition = t.transform.position;
                        var botRotation = t.transform.eulerAngles;
                        var botVelocity = t.GetComponent<Rigidbody>().velocity;
                        var botAngular = t.GetComponent<Rigidbody>().angularVelocity;
                        BotProperties.Add(new[,]{
                            {botPosition.x, botPosition.y, botPosition.z},
                            {botRotation.x, botRotation.y, botRotation.z},
                            {botVelocity.x, botVelocity.y, botVelocity.z},
                            {botAngular.x, botAngular.y, botAngular.z}
                        });
                        botScores.Add(t.GetComponent<BaseController>().Score);
                        botLaps.Add(t.GetComponent<BaseController>().CurrentLap);
                        botCurrentGas.Add(t.GetComponent<BaseController>().CurrentGasPedalAmount);
                        botCurrentSteering.Add(t.GetComponent<BaseController>().CurrentSteeringAmount);
                        botLastWayPoints.Add(t.GetComponent<BaseController>()._lastWaypointCrossed);
                        botCurrentWayPoints.Add(t.GetComponent<BaseController>()._currentWaypoint);
                        botCurrentAIPoint.Add(t.GetComponent<VehicleAI>()._currentWaypoint);

                        botPrevious.Add(new float[] { t.GetComponent<SolidController>().previousPos[0], t.GetComponent<SolidController>().previousPos[1], t.GetComponent<SolidController>().previousPos[2] });
                        botPrevEuler.Add(new float[] { t.GetComponent<SolidController>().previousEurler[0], t.GetComponent<SolidController>().previousEurler[1], t.GetComponent<SolidController>().previousEurler[2] });

                        List<float[]> theseWheels = new List<float[]>();
                        foreach (SolidWheelBehaviour wheel in t.GetComponent<VehicleAI>().solidWheels)
                        {
                            theseWheels.Add(wheel.GetProperties());
                        }
                        botWheels.Add(theseWheels);
                    }
                }
            
                currentSteering = _player.GetComponent<BaseController>().CurrentSteeringAmount;
                currentGas = _player.GetComponent<BaseController>().CurrentGasPedalAmount;
                lastWayPoint = _player.GetComponent<BaseController>()._lastWaypointCrossed;
                currentPoint = _player.GetComponent<GoBlendSolidController>().currentPoint;
                bestScore = _player.GetComponent<GoBlendSolidController>().bestScore;

                explorePoints = new List<float[]>();
                rotation = new List<float[]>();
                velocities = new List<float[]>();
                angulars = new List<float[]>();

                foreach (var point in _player.GetComponent<GoBlendSolidController>()?.ExplorePoints)
                {
                    explorePoints.Add(new [] {point[0], point[1], point[2]});
                }
                foreach (var point in _player.GetComponent<GoBlendSolidController>()?.Rotations)
                {
                    rotation.Add(new [] { point[0], point[1], point[2] });
                }
                foreach (var point in _player.GetComponent<GoBlendSolidController>()?.Velocities)
                {
                    velocities.Add(new [] { point[0], point[1], point[2] });
                }
                foreach (var point in _player.GetComponent<GoBlendSolidController>()?.Angulars)
                {
                    angulars.Add(new [] { point[0], point[1], point[2] });
                }

                wheels = new List<float[]>();
                foreach (SolidWheelBehaviour wheel in _player.GetComponent<GoBlendSolidController>().solidWheels)
                {
                    wheels.Add(wheel.GetProperties());
                }
                previousPos = new float[] { _player.GetComponent<SolidController>().previousPos[0], _player.GetComponent<SolidController>().previousPos[1], _player.GetComponent<SolidController>().previousPos[2] };
                previousEuler = new float[] { _player.GetComponent<SolidController>().previousEurler[0], _player.GetComponent<SolidController>().previousEurler[1], _player.GetComponent<SolidController>().previousEurler[2] };
                cell = backEnd?.CurrentCell;
            }
        }
    }
}
