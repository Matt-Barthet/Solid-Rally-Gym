using System;
using System.Collections.Generic;
using System.Linq;
using Go_Blend.Scripts.Utilities;
using MoreMountains.HighroadEngine;
using UnityEngine;

namespace Pagan_Experiment
{
    public class DataLogger : MonoBehaviour {
        
        private GameObject player;
        public RaceManager raceManager;
        private List<int> playerStanding;
        private List<int> playerScore;
        private List<float> playerSpeed;
        private List<int> playerIsGrounded;
        private List<int> playerIsLooping;
        private List<int> playerIsOffRoad;
        private List<int> playerIsCrashing;
        private List<float> playerGasPedal;
        private List<float> playerSteering;
        private List<int> playerLap;
        private List<float> playerDistanceToWayPoint;
        private List<float> playerDeltaDistance;
        private List<float> playerDeltaRotation;
        public List<int> playerRespawn;
        private List<int> visibleBotCount;
        private List<int> botStanding;
        private List<int> botScore;
        private List<float> botSpeed;
        private List<int> botIsGrounded;
        private List<int> botIsLooping;
        private List<int> botIsOffRoad;
        private List<int> botIsCrashing;
        private List<float> botGasPedal;
        private List<float> botSteering;
        private List<int> botLap;
        private List<float> botDistanceToWayPoint;
        private List<float> botDeltaDistance;
        private List<float> botDeltaRotation;
        private List<float> botPlayerDistance;
        public List<float> botRespawn;
        private List<int> visibleLoopCount;
        private GameObject[] allBots;
        private List<GameObject> visibleBots;
        private GameObject[] allLoop;
        private List<GameObject> visibleLoop;
        public LayerMask botLayer;

        private void Awake() {
            ResetForm();
        }
    
        private void Start() {
            player = GameObject.FindGameObjectWithTag("Player");
        }
    
        private void FixedUpdate() {
            visibleBots = new List<GameObject>();
            allBots = GameObject.FindGameObjectsWithTag("Enemy");
            // player = GameObject.FindGameObjectWithTag("Player");
            foreach (GameObject character in allBots) {
                if (raceManager.game == "solid") {
                    //if (character.transform.GetComponentInChildren<MeshRenderer>().isVisible) {
                    if (player != null) {
                        RaycastHit hit;
                        Vector3 rayDir = (character.transform.position - player.transform.position).normalized;
                        if (Physics.Raycast(player.transform.position, rayDir, out hit, Mathf.Infinity, botLayer)) {
                            // Debug.DrawRay(player.transform.position, rayDir * hit.distance, Color.green);
                            if (hit.collider.gameObject == character) {
                                visibleBots.Add(character);
                            }
                        }
                    }
                }
            }
            visibleLoop = new List<GameObject>();
            allLoop = GameObject.FindGameObjectsWithTag("LoopZone");
            foreach (GameObject loop in allLoop) {
                if (raceManager.game == "solid") {
                    if (player != null) {
                        //if (loop.transform.GetChild(2).GetComponent<MeshRenderer>().isVisible) {
                        RaycastHit hit;
                        Vector3 rayDir = (new Vector3(loop.transform.position.x, loop.transform.position.y+22, loop.transform.position.z)
                                          - player.transform.position).normalized;
                        if (Physics.Raycast(player.transform.position, rayDir, out hit)) {
                            // Debug.DrawRay(player.transform.position, rayDir * hit.distance, Color.green);
                            if (hit.collider.gameObject == loop) {
                                visibleLoop.Add(loop);
                            }
                        } else if (player.GetComponent<BaseController>().IsInLoopZone) {
                            visibleLoop.Add(loop);
                        }
                    }
                }
            }
            UpdateForm();
        }

        private void UpdateForm() {
            player = GameObject.FindGameObjectWithTag("Player");
            var player_car = player.GetComponent<SolidController>();
            playerStanding.Add(player_car._standing);
            playerScore.Add(player_car.Score);
            playerSpeed.Add(Mathf.Abs(player_car.Speed));
            playerIsGrounded.Add(player_car.IsGrounded ? 1 : 0);
            playerIsLooping.Add(player_car.IsInLoopZone ? 1 : 0);
            if (raceManager.game == "solid") {
                playerIsCrashing.Add(player_car.GetComponent<SolidSoundBehaviour>().isCrashing ? 1 : 0);
                playerIsOffRoad.Add(player_car.GetComponent<SolidController>().isOffRoad ? 1 : 0);
            }
            playerGasPedal.Add(player_car.CurrentGasPedalAmount);
            playerSteering.Add(player_car.CurrentSteeringAmount);
            playerLap.Add(player_car.CurrentLap);
            playerDistanceToWayPoint.Add(player_car.DistanceToNextWaypoint);
            playerDeltaDistance.Add(player_car.DeltaDistance);
            playerDeltaRotation.Add(player_car.DeltaRotation);
            for (int i = 0; i < visibleBots.Count; i++) {
                var car = visibleBots[i].GetComponent<SolidController>();
                botStanding.Add(car._standing);
                botScore.Add(car.Score);
                botSpeed.Add(Mathf.Abs(car.Speed));
                botIsGrounded.Add(car.IsGrounded ? 1 : 0);
                botIsLooping.Add(car.IsInLoopZone ? 1 : 0);
                if (raceManager.game == "solid") {
                    botIsCrashing.Add(car.GetComponent<SolidSoundBehaviour>().isCrashing ? 1 : 0);
                    botIsOffRoad.Add(car.GetComponent<SolidController>().isOffRoad ? 1 : 0);
                }
                botGasPedal.Add(car.CurrentGasPedalAmount);
                botSteering.Add(car.CurrentSteeringAmount);
                botLap.Add(car.CurrentLap);
                botDistanceToWayPoint.Add(car.DistanceToNextWaypoint);
                botDeltaDistance.Add(car.DeltaDistance);
                botPlayerDistance.Add(car.DeltaPlayerDistance);
                botDeltaRotation.Add(car.DeltaRotation);
            }
            visibleBotCount.Add(visibleBots.Count);
            visibleLoopCount.Add(visibleLoop.Count);
        }

        public float[] GetSurrogateVector()
        {
            var vector = new double[32];
            vector[0] = playerStanding.DefaultIfEmpty(0).Average();
            vector[1] = playerScore.DefaultIfEmpty(0).Average();
            vector[2] = playerSpeed.DefaultIfEmpty(0).Average();
            vector[3] = playerIsGrounded.DefaultIfEmpty(0).Average();
            vector[4] = 1 - playerIsGrounded.DefaultIfEmpty(0).Average();
            vector[5] = playerIsLooping.DefaultIfEmpty(0).Average();
            vector[6] = playerIsCrashing.DefaultIfEmpty(0).Average();
            vector[7] = playerIsOffRoad.DefaultIfEmpty(0).Average();
            vector[8] = playerGasPedal.DefaultIfEmpty(0).Average();
            vector[9] = playerSteering.DefaultIfEmpty(0).Average();
            vector[10] = playerLap.DefaultIfEmpty(0).Average();
            vector[11] = playerDistanceToWayPoint.DefaultIfEmpty(0).Average();
            vector[12] = playerDeltaDistance.DefaultIfEmpty(0).Average();
            vector[13] = playerDeltaRotation.DefaultIfEmpty(0).Average();
            vector[14] = playerRespawn.DefaultIfEmpty(0).Average();
            vector[15] = visibleBotCount.DefaultIfEmpty(0).Average();
            vector[16] = botStanding.DefaultIfEmpty(0).Average();
            vector[17] = botScore.DefaultIfEmpty(0).Average();
            vector[18] = botSpeed.DefaultIfEmpty(0).Average();
            vector[19] = botIsGrounded.DefaultIfEmpty(0).Average();
            vector[20] = botIsLooping.DefaultIfEmpty(0).Average();
            vector[21] = botIsOffRoad.DefaultIfEmpty(0).Average();
            vector[22] = botIsCrashing.DefaultIfEmpty(0).Average();
            vector[23] = botGasPedal.DefaultIfEmpty(0).Average();
            vector[24] = botSteering.DefaultIfEmpty(0).Average();
            vector[25] = botLap.DefaultIfEmpty(0).Average();
            vector[26] = botDistanceToWayPoint.DefaultIfEmpty(0).Average();
            vector[27] = botDeltaDistance.DefaultIfEmpty(0).Average();
            vector[28] = botDeltaRotation.DefaultIfEmpty(0).Average();
            vector[29] = botPlayerDistance.DefaultIfEmpty(0).Average();
            vector[30] = botRespawn.DefaultIfEmpty(0).Average();
            vector[31] = visibleLoopCount.DefaultIfEmpty(0).Average();
            var surrogateVector = new float[32];
            for (var i = 0; i < vector.Length; i++)
            {
                surrogateVector[i] = Utility.MinMaxScaling((float) vector[i], HumanModel.Minimums[i], HumanModel.Maximums[i]);
                surrogateVector[i] = (float) Math.Round(surrogateVector[i], 2);
            }
            ResetForm();
            return surrogateVector;
        }

        private void ResetForm() {
            playerStanding = new List<int>();
            playerScore = new List<int>();
            playerSpeed = new List<float>();
            playerIsGrounded = new List<int>();
            playerIsLooping = new List<int>();
            playerIsOffRoad = new List<int>();
            playerIsCrashing = new List<int>();
            playerGasPedal = new List<float>();
            playerSteering = new List<float>();
            playerLap = new List<int>();
            playerDistanceToWayPoint = new List<float>();
            playerDeltaDistance = new List<float>();
            playerDeltaRotation = new List<float>();
            playerRespawn = new List<int>();
            visibleBotCount = new List<int>();
            botStanding = new List<int>();
            botScore = new List<int>();
            botSpeed = new List<float>();
            botIsGrounded = new List<int>();
            botIsLooping = new List<int>();
            botIsOffRoad = new List<int>();
            botIsCrashing = new List<int>();
            botGasPedal = new List<float>();
            botSteering = new List<float>();
            botLap = new List<int>();
            botDistanceToWayPoint = new List<float>();
            botDeltaDistance = new List<float>();
            botDeltaRotation = new List<float>();
            botPlayerDistance = new List<float>();
            botRespawn = new List<float>();
            visibleLoopCount = new List<int>();
        }
    }
}
