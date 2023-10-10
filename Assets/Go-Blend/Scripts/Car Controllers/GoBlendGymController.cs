using System.Collections.Generic;
using System.IO;
using System.Linq;
using Go_Blend.Scripts.Car_Controllers.Base_Classes;
using Go_Blend.Scripts.Utilities;
using MoreMountains.HighroadEngine;
using Pagan_Experiment;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.SideChannels;
using UnityEngine;
using Directory = System.IO.Directory;
using Input = UnityEngine.Input;
using Quaternion = UnityEngine.Quaternion;
using Vector3 = UnityEngine.Vector3;

namespace Go_Blend.Scripts.Car_Controllers
{
    public class GoBlendGymController : Agent
    {
        private GoBlendSolidController goBlendSolidController;
        private SolidController solidController;
        private Rigidbody rigidBody;
        private DataLogger logger;
        private MySideChannel mySideChannel;
        private List<float []> raycasts;
        private List<int> scores;
        private List<float> steering, pedals;
        private float targetSteering, targetPedal;
        public bool saveCells, generateArousal;
        
        private void Start()
        {
            if (!isActiveAndEnabled) return;
            
            // HumanModel.Init("Cluster2", 5);
            
            // Create the directory to store the data from the unity side.
            var counter = Directory.GetDirectories($"{Application.persistentDataPath}", "*", SearchOption.TopDirectoryOnly).Length - 2;
            Directory.CreateDirectory($"{Application.persistentDataPath}/Run_{counter}");
            Directory.CreateDirectory($"{Application.persistentDataPath}/Run_{counter}/Cells");
            StateSaveLoad.SetCurrentPath($"{Application.persistentDataPath}/Run_{counter}");

            logger = GameObject.FindWithTag("DataLogger").GetComponent<DataLogger>();
            solidController = GetComponent<SolidController>();
            goBlendSolidController = GetComponent<GoBlendSolidController>();
            rigidBody = GetComponent<Rigidbody>();
            
            mySideChannel = new MySideChannel(this, rigidBody);
            SideChannelManager.RegisterSideChannel(mySideChannel);
            Time.fixedDeltaTime = 0.02f;

            raycasts = new List<float []>();
            scores = new List<int>();
            steering = new List<float>();
            pedals = new List<float>();
        }
        
        public override void OnEpisodeBegin()
        {
            if (!isActiveAndEnabled) return;
            raycasts.Clear();
            steering.Clear();
            pedals.Clear();
            previousScore = 0;
            StateSaveLoad.Reset();
        }
        
        public override void CollectObservations(VectorSensor sensor)
        {

            /*-------------------------Low Resolution States-----------------------------*/
            var message = new OutgoingMessage();
            var lowResStates = goBlendSolidController.GetState()[0];
            var messageToPass = "[Low-Resolution State]:" +
                                     $"Speed:{lowResStates[0]}," +
                                     $"Segment:{lowResStates[1]}," +
                                     $"Sub-Segment:{lowResStates[2]}," +
                                     $"Bot Nearby:{lowResStates[3]}," +
                                     $"Lap Number:{lowResStates[4]}," +
                                     $"Rotation:{lowResStates[5]}";
            message.WriteString(messageToPass);
            mySideChannel.SendMessage(message);

            /*---------------------------------------------------------------------------*/
            var velocity = rigidBody.velocity;
            sensor.AddObservation(velocity.x);
            sensor.AddObservation(velocity.y);
            sensor.AddObservation(velocity.z);
            
            var rotation = transform.eulerAngles;
            sensor.AddObservation(rotation.x);
            sensor.AddObservation(rotation.y);
            sensor.AddObservation(rotation.z);

            /*if (generateArousal)
            {
                var arousal = HumanModel.GetClosestHuman(logger.GetSurrogateVector());
                sensor.AddObservation(arousal);
            }
            else
            {
                sensor.AddObservation(0);
                sensor.AddObservation(0);
            }*/
            
            var startingVec = Vector3.forward;
            var numberOfRaycasts = 36;
            var maxDistance = 100;    
            var layerMask = ~LayerMask.GetMask("Ignore Raycast");

            var raycast = new float[numberOfRaycasts];
            for (var i = 0; i < numberOfRaycasts; i++)
            {
                var direction = transform.TransformDirection(Quaternion.Euler(0, -360 / numberOfRaycasts * i, 0) * startingVec);
                if (Physics.Raycast(transform.position, direction, out var hit, maxDistance, layerMask))
                {
                    Debug.DrawRay(transform.position, direction * hit.distance, Color.white);
                    var value = hit.distance;
                    sensor.AddObservation(value);
                    raycast[i] = value;
                }
                else
                {
                    sensor.AddObservation(maxDistance);
                    raycast[i] = maxDistance;
                }
            }

            var nextCheckpoint = goBlendSolidController.explorePoints[goBlendSolidController.currentPoint];
            var toCheckpoint = (nextCheckpoint.position - transform.position).normalized;
            var angle = Vector3.Angle(transform.forward, toCheckpoint);
            
            sensor.AddObservation(solidController.CurrentSteeringAmount);
            sensor.AddObservation(solidController.CurrentGasPedalAmount);
            sensor.AddObservation(solidController.isOffRoad);
            sensor.AddObservation(solidController.IsInLoopZone);
            sensor.AddObservation(solidController.isJumping);
            sensor.AddObservation(Vector3.Distance(nextCheckpoint.position, transform.position));
            sensor.AddObservation(angle);

            if (saveCells)
            {
                var angular = rigidBody.angularVelocity;
                goBlendSolidController.Rotations.Add(new [] { rotation.x, rotation.y, rotation.z });
                goBlendSolidController.Velocities.Add(new [] { velocity.x, velocity.y, velocity.z });
                goBlendSolidController.Angulars.Add(new [] { angular.x, angular.y, angular.z });
                raycasts.Add(raycast);
                scores.Add(solidController.Score);
            }
            
            var dirMessage = new OutgoingMessage();
            // dirMessage.WriteString($"[Direction]:{transform.forward.x},{transform.forward.y},{transform.forward.z}");
            mySideChannel.SendMessage(dirMessage);
        }
        
        private float fValue;
        private const float Sensitivity = 3f;
        private const float Dead = 0.001f;
        
        private float SmoothSteering(int target)
        {
            var sign = Mathf.Sign(fValue);
            if (Mathf.Sign(target) != sign && target != 0)
            {
                fValue = 0f;
            }
            fValue = Mathf.MoveTowards(fValue, target, Sensitivity * Time.deltaTime);
            if (Mathf.Abs(fValue) < Dead)
            {
                fValue = 0f;
            }
            return fValue;
        }

        private int previousScore;
        
        public override void OnActionReceived(ActionBuffers actionBuffers)
        {

#if UNITY_EDITOR
            return;
#endif
            targetSteering = SmoothSteering(actionBuffers.DiscreteActions[0]);
            targetPedal = actionBuffers.DiscreteActions[1];
            SetReward(solidController.Score);          

            /*
            var targetDir = goBlendSolidController.explorePoints[goBlendSolidController.currentPoint].transform.position - transform.position;
            targetDir = targetDir.normalized;
            var dot = Vector3.Dot(targetDir, transform.forward);
            var angle = Mathf.Acos( dot ) * Mathf.Rad2Deg;
            float newReward = 0;
            if (angle > 90)
            {
                // newReward = -0.1f;
            }
            else if (goBlendSolidController.currentPoint > previousScore)
            {
                newReward = 1;
                previousScore = goBlendSolidController.currentPoint;
            }
            SetReward(newReward); 
            var code = actionBuffers.DiscreteActions[0];
            var currentSteeringInput = 0;
            targetPedal = 0;
            targetSteering = 0;
            switch (code)
            {
                case 1:
                    targetPedal = 1;
                    break;
                case 2:
                    targetPedal = -1;
                    break;
                case 3:
                    targetPedal = 1;
                    currentSteeringInput = -1;
                    break;
                case 4:
                    targetPedal = 1;
                    currentSteeringInput = 1;
                    break;
                case 5:
                    targetPedal = -1;
                    currentSteeringInput = -1;
                    break;
                case 6:
                    targetPedal = -1;
                    currentSteeringInput = 1;
                    break;
                case 7:
                    currentSteeringInput = -1;
                    break;
                case 8:
                    currentSteeringInput = 1;
                    break;
            }
            targetSteering = SmoothSteering(currentSteeringInput); 
             if (!saveCells) return;
            pedals.Add(targetPedal);
            steering.Add(targetSteering);
            StateSaveLoad.SaveEnvironment($"Time-step-{scores.Count}");
            if (solidController.Score != 8) return;
            TextWriter tw = new StreamWriter($"{StateSaveLoad.CurrentPath}/Human_Trace.csv");
            tw.Write("Position_X,Position_Y,Position_Z,Velocity_X,Velocity_Y,Velocity_Z,Rotation_X,Rotation_Y,Rotation_Z,Raycast_1,Raycast_2,Raycast_3,Raycast_4,Raycast_5,Raycast_6,Raycast_7,Raycast_8,Score,Pedal,Steering\n");
            for (var i = 0; i < goBlendSolidController.ExplorePoints.Count; i++)
            {
                var casts = raycasts[i].Aggregate("", (current, cast) => current + $",{cast}");
                tw.WriteLine($"{goBlendSolidController.ExplorePoints[i][0]},{goBlendSolidController.ExplorePoints[i][1]},{goBlendSolidController.ExplorePoints[i][2]}," +
                             $"{goBlendSolidController.Velocities[i][0]},{goBlendSolidController.Velocities[i][1]},{goBlendSolidController.Velocities[i][2]}," +
                             $"{goBlendSolidController.Rotations[i][0]},{goBlendSolidController.Rotations[i][1]},{goBlendSolidController.Rotations[i][2]}" + //Leave out comma cos of above for loop.
                             $"{casts}," +
                             $"{scores[i]}," +
                             $"{pedals[i]},{steering[i]}"
                );
            }
            tw.Close();
            DestroyImmediate(gameObject);*/
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            targetSteering = Input.GetAxis("Horizontal");
            targetPedal = (int) Input.GetAxisRaw("Vertical");
        }

        protected void FixedUpdate()
        {
            solidController.VerticalPosition(targetPedal);
            solidController.HorizontalPosition(targetSteering);
        }

        private void OnDestroy()
        {
            // HumanModel.SaveDict();
        }
    }
}
