using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Go_Blend.Scripts.Utilities;
using MoreMountains.HighroadEngine;
using UnityEngine;
using UnityEngine.Serialization;
using UnityEngine.UIElements;

namespace Go_Blend.Scripts.Car_Controllers.Base_Classes
{
    public class GoBlendSolidController : GoBlendBaseController
    {
        [HideInInspector]
        public SolidWheelBehaviour[] solidWheels;
        protected SolidController SolidController;

        [FormerlySerializedAs("_explorePoints")] [HideInInspector]
        public List<Transform> explorePoints;

        [FormerlySerializedAs("CurrentPoint")] [HideInInspector]
        public int currentPoint;
        
        private const int MaxLaps = 2;
        protected float[] _steeringPositions, _pedalPositions;

        [HideInInspector]
        public float currentTime;

        public float bestScore;

        private Vector3 startingPos, startingVel, startingAngular;
        private Quaternion startingRot;
        
        protected new void Start()
        {
            base.Start();
            solidWheels = GetComponentsInChildren<SolidWheelBehaviour>();
            RigidBody = GetComponent<Rigidbody>();
            SolidController = GetComponent<SolidController>();
            
            var explorePoint = GameObject.FindGameObjectWithTag("ExplorePointContainer");
            explorePoints = new List<Transform>();
            foreach (Transform child in explorePoint.transform)
            {
                explorePoints.Add(child);
            }
            
            startingPos = transform.position;
            startingRot = transform.rotation;
            startingVel = Vector3.zero;
            startingAngular = Vector3.zero;
            
            ExplorePoints = new List<float[]>();
            Rotations = new List<float[]>();
            Angulars = new List<float[]>();
            Velocities = new List<float[]>();
            Initialize();
        }
        
        public new void Initialize()
        {
            base.Initialize();
            SolidController.CurrentSteeringAmount = 0;
            SolidController.CurrentGasPedalAmount = 0;
            SolidController.previousPos = transform.position;
            SolidController.previousEurler = transform.eulerAngles;
            SolidController.CurrentLap = 0;
            SolidController.Score = 0;
            SolidController._currentWaypoint = 0;
            SolidController._lastWaypointCrossed = -1;
            currentPoint = 0;
            bestScore = 0;
            ExplorePoints = new List<float[]>();
            Rotations = new List<float[]>();
            Angulars = new List<float[]>();
            Velocities = new List<float[]>();
            transform.SetPositionAndRotation(startingPos, startingRot);
            RigidBody.velocity = startingVel;
            RigidBody.angularVelocity = startingAngular;
            foreach (var wheel in solidWheels) wheel.Reset();
        }
        
        public float DistanceScore()
        {
            var currentCheckpoint = explorePoints[currentPoint].position;
            var previousCheckpoint = currentPoint == 0 ? new Vector3(-76, 1.4f, 32) : explorePoints[currentPoint - 1].position;
            var maxDistance = Vector3.Distance(previousCheckpoint, currentCheckpoint);
            return 1 - Vector3.Distance(transform.position, currentCheckpoint) / maxDistance;
        }
        
        public override bool GetFinal()
        {
            return SolidController.CurrentLap >= MaxLaps;
        }

        public override float GetScore()
        {
            return SolidController.Score;
        }

        private void OnTriggerEnter(Collider collision)
        {
            if (!isActiveAndEnabled) return;
            if (collision.transform == explorePoints[currentPoint])
                currentPoint += 1;
            currentPoint %= explorePoints.Count;
        }
        
        public float CurrentWayPoint()
        {
            return SolidController.CurrentLap * explorePoints.Count + currentPoint; 
        }
        
        public override float BehaviorReward()
        {
            return Utility.MinMaxScaling(GetScore(), 0, 8 * MaxLaps);
        }
        
        
        public override float[][] GetInputs(bool expert)
        {
            if (expert)
            {
                _steeringPositions = new float[] { 0, 0, 0, 0, 1, 1, 1, -1, -1, -1 };
                _pedalPositions = new float[] { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1 };
            } else
            {
                _steeringPositions = new float[] { -1, 0, 1};
                _pedalPositions = new float[] { -1, 0, 1 };
            }
            return new[] { _steeringPositions, _pedalPositions };
        }

        public override string[][] GetState()
        {
            var newState = new string[6];
            var properties = new string[4];
            
            if (SolidController.Speed < 30) newState[0] = "Slow";
            else newState[0] = "Fast";

            newState[3] = "False";
            newState[4] = SolidController.CurrentLap.ToString();

            var trans = transform;
            var position = trans.position;
            var eulerAngles = trans.eulerAngles;
            var velocity = RigidBody.velocity;
            var angular = RigidBody.angularVelocity;
            
            for (var i = 1; i < 7; i++)
            {
                if (Vector3.Angle(trans.forward, explorePoints[currentPoint].position - trans.position) <= i * 30)
                {
                    newState[5] = (i * 30).ToString(CultureInfo.InvariantCulture);
                    break;
                }
            }
            
            properties[0] = $"{position.x}|{position.y}|{position.z}";
            properties[1] = $"{eulerAngles.x}|{eulerAngles.y}|{eulerAngles.z}";
            properties[2] = $"{velocity.x}|{velocity.y}|{velocity.z}";
            properties[3] = $"{angular.x}|{angular.y}|{angular.z}";

            try
            {
                newState[1] = SolidController._groundGameObject.transform.parent.name;
                newState[2] = SolidController._groundGameObject.name;
                if (SolidController._raceManager.bots.Any(bot =>
                        bot.GetComponent<SolidController>()._groundGameObject.transform.parent ==
                        SolidController._groundGameObject.transform.parent))
                    newState[3] = "True";
            }
            catch (Exception)
            {
                newState[1] = "Airborne";
                newState[2] = "Airborne";
            }
            
            return new[] { newState, properties };
        }
        
    }
}
