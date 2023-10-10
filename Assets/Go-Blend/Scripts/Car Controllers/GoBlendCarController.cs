using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using Go_Blend.Scripts.Car_Controllers.Base_Classes;
using Go_Blend.Scripts.Utilities;
using MoreMountains.HighroadEngine;
using UnityEditor;
using UnityEngine;

namespace Go_Blend.Scripts.Car_Controllers
{
    public class GoBlendCarController : GoBlendSolidController
    {
        private const float Penalty = 0.25f;
        private const float OptimalTimePerPoint = 4.75f;

        private List<float[]> replayList;
        private Vector3 startingPos, startingVel, startingAngular;
        private Quaternion startingRot;
        private bool replayEnded, tookFirst;
    
        private new void Start()
        {
            solidWheels = GetComponentsInChildren<SolidWheelBehaviour>();
            RigidBody = GetComponent<Rigidbody>();
            SolidController = GetComponent<SolidController>();
            
            var explorePoint = GameObject.FindGameObjectWithTag("ExplorePointContainer");
            explorePoints = new List<Transform>();
            foreach (Transform child in explorePoint.transform)
            {
                explorePoints.Add(child);
            }
            
            Time.captureFramerate = framerate;
            Time.fixedDeltaTime = 0.025f;

            startingPos = transform.position;
            startingRot = transform.rotation;
            startingVel = Vector3.zero;
            startingAngular = Vector3.zero;
            backEnd.Start();
            base.Start();
            Initialize();
        }
        
        public new void Initialize()
        {
            if (!isActiveAndEnabled) return;

            base.Initialize();
            replayEnded = false;
            replayList = backEnd.ReplayActions?.ToList() ?? new List<float[]>();
        
            Time.captureFramerate = framerate;
            currentTime = 0f;
        
            transform.SetPositionAndRotation(startingPos, startingRot);
            RigidBody.velocity = startingVel;
            RigidBody.angularVelocity = startingAngular;
        
            SolidController.CurrentSteeringAmount = 0;
            SolidController.CurrentGasPedalAmount = 0;
            SolidController.previousPos = transform.position;
            SolidController.previousEurler = transform.eulerAngles;
            SolidController.CurrentLap = 0;
            SolidController.Score = 0;
            SolidController._currentWaypoint = 0;
            SolidController._lastWaypointCrossed = -1;
            currentPoint = 0;
            foreach (var wheel in solidWheels) wheel.Reset();
            Resources.UnloadUnusedAssets();
        }

        private void ReplayAction()
        {
            var replay = replayList[0];
            ReplayPoints.Add(GoExplore.VectorToFloat(GetPosition()));
            SolidController.VerticalPosition(replay[1]);
            targetSteering = replay[0];
            CalculateDeltas();
            replayList.RemoveAt(0);
        }

        private float targetSteering; 
    
        private void ExploreAction()
        {
            if (tookFirst)
            {
                if (backEnd.CurrentCell != null) backEnd.AssessCell();
                if (backEnd.returning) return;
            }
            var action = new[]
            {
                _steeringPositions[UnityEngine.Random.Range(0, _steeringPositions.Length)],
                _pedalPositions[UnityEngine.Random.Range(0, _pedalPositions.Length)],
            };
            targetSteering = action[0];
            SolidController.VerticalPosition(action[1]);
            backEnd.ConstructCell(GetState(), action);
            ExplorePoints = backEnd.currentCoordinates;
            tookFirst = true;
        }

        protected void FixedUpdate()
        {
            if (!backEnd.exploring) return;
        
            if (backEnd.returning)
            {
                backEnd.returning = false;
                Restart = true;
                return;
            }
        
            currentTime += Time.fixedDeltaTime;
            SolidController.HorizontalPosition(Mathf.MoveTowards(SolidController.CurrentSteeringAmount, targetSteering, 0.1f));

            if (Math.Round(currentTime, 3) < 0.25f) return;

            if (replayList.Count > 0) ReplayAction();
            else
            {
                if (!replayEnded)
                {
                    /*backEnd.currentCoordinates = new List<float[]>(replayPoints);
                backEnd.currentRotations = new List<float[]>(rotations);
                backEnd.currentVelocities = new List<float[]>(velocities);
                backEnd.currentAngulars = new List<float[]>(angulars);*/
                    replayEnded = true;
                }
                ExploreAction();
            }
            currentTime = 0;
        }

        private void LateUpdate()
        {
            if (!Restart) return;
            Restart = false;
            tookFirst = false;
            StateSaveLoad.Reset();
        }

        private float DistanceScore()
        {
            var currentCheckpoint = explorePoints[currentPoint].position;
            var previousCheckpoint = currentPoint == 0 ? new Vector3(-76, 1.387f, 32) : explorePoints[currentPoint - 1].position;
            var maxDistance = Vector3.Distance(previousCheckpoint, currentCheckpoint);
            return 1 - Vector3.Distance(transform.position, currentCheckpoint) / maxDistance;
        }

        private float SpeedPenalty()
        {
            var optimalTime = OptimalTimePerPoint * (SolidController.CurrentLap * explorePoints.Count + currentPoint + 1);
            var currentTime = backEnd.currentActionTrajectory.Count;
            if (currentTime <= optimalTime) return 0;
            return -Penalty * (currentTime - optimalTime);
        }

        public void DrawLine(Vector3 p1, Vector3 p2, float width)
        {
            int count = 1 + Mathf.CeilToInt(width); // how many lines are needed.
            if (count == 1)
            {
                Gizmos.DrawLine(p1, p2);
            }
            else
            {
                Camera c = Camera.current;
                if (c == null)
                {
                    Debug.LogError("Camera.current is null");
                    return;
                }
                var scp1 = c.WorldToScreenPoint(p1);
                var scp2 = c.WorldToScreenPoint(p2);
    
                Vector3 v1 = (scp2 - scp1).normalized; // line direction
                Vector3 n = Vector3.Cross(v1, Vector3.forward); // normal vector
    
                for (int i = 0; i < count; i++)
                {
                    Vector3 o = 0.99f * n * width * ((float)i / (count - 1) - 0.5f);
                    Vector3 origin = c.ScreenToWorldPoint(scp1 + o);
                    Vector3 destiny = c.ScreenToWorldPoint(scp2 + o);
                    Gizmos.DrawLine(origin, destiny);
                }
            }
        }

        private void OnDrawGizmos()
        {
            if (!isActiveAndEnabled) return;
#if UNITY_EDITOR
            if (!Application.isPlaying) return;
            // DrawArousal();
            DrawDeltas();
#endif
        }

        private void DrawDeltas()
        {
#if UNITY_EDITOR
            Gizmos.color = Color.blue;
            for (var i = 0; i < ExplorePoints.Count; i++)
            {
                Gizmos.DrawSphere(FloatToVector(ExplorePoints[i]), 1f);
                if (i > 0)
                    DrawLine(FloatToVector(ExplorePoints[i]), FloatToVector(ExplorePoints[i - 1]), 3);
            }

            for (var i = 0; i < ReplayPoints.Count; i++)
            {
                Gizmos.color = Color.red;
                Gizmos.DrawSphere(FloatToVector(ReplayPoints[i]), 1f);
                if (i > 0)
                    DrawLine(FloatToVector(ReplayPoints[i]), FloatToVector(ReplayPoints[i - 1]), 3);

                Gizmos.color = Color.white;
                var distance = Vector3.Distance(FloatToVector(ReplayPoints[i]), FloatToVector(ExplorePoints[i]));
                DrawLine(FloatToVector(ReplayPoints[i]), FloatToVector(ExplorePoints[i]), 3);
                Handles.Label(FloatToVector(ReplayPoints[i]) + new Vector3(0, 2, 0), distance.ToString());
            }
#endif
        }

        private void DrawArousal()
        {
#if UNITY_EDITOR
            var arousalValues = backEnd.currentArousalTrace;    
            for (var i = 0; i < ExplorePoints.Count - 1; i++)
            {
                try
                {
                    if (arousalValues[i] < 0.5)
                    {
                        Gizmos.color = Color.Lerp(Color.blue * 1.5f, Color.white, arousalValues[i] * 2);
                    }
                    else
                    {
                        Gizmos.color = Color.Lerp(Color.white, Color.red * 1.5f, Utility.MinMaxScaling(arousalValues[i], 0.5f, 1));
                    }

                    Gizmos.DrawSphere(FloatToVector(ExplorePoints[i]), 2f);

                    if (i > 0)
                        DrawLine(FloatToVector(ExplorePoints[i]), FloatToVector(ExplorePoints[i - 1]), 3);
                }
                catch (ArgumentOutOfRangeException) { }
            }
#endif
        }

    }
}
