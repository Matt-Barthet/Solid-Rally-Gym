using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

namespace Go_Blend.Scripts.Car_Controllers.Base_Classes
{
    public abstract class GoBlendBaseController : MonoBehaviour
    {
        public GoExplore backEnd;
        protected Rigidbody RigidBody;

        private Text archiveSize, trajectorySize, trajectoryScore, bestScoreUI, bestSizeUI, iterations, bestGameScoreUI;
        public List<float[]> ExplorePoints, Rotations, Velocities, Angulars;
        protected List<float[]> ReplayPoints;

        protected bool Restart = false;

        [HideInInspector]
        public bool legalState;
        
        public int framerate = 5;

        protected void Start()
        {
            GoExplore.PlayerController = this;
            archiveSize = GameObject.FindGameObjectWithTag("ArchiveSize")?.GetComponent<Text>();
            trajectorySize = GameObject.FindGameObjectWithTag("TrajectorySize")?.GetComponent<Text>();
            trajectoryScore = GameObject.FindGameObjectWithTag("TrajectoryScore")?.GetComponent<Text>();
            bestScoreUI = GameObject.FindGameObjectWithTag("BestScore")?.GetComponent<Text>();
            bestSizeUI = GameObject.FindGameObjectWithTag("BestSize")?.GetComponent<Text>();
            iterations = GameObject.FindGameObjectWithTag("Iterations")?.GetComponent<Text>();
            bestGameScoreUI = GameObject.FindGameObjectWithTag("BestInGame")?.GetComponent<Text>();
            ReplayPoints = new List<float[]>();
        }

        protected void Initialize()
        {
            ExplorePoints = backEnd.currentCoordinates;
            Rotations = backEnd.currentRotations;
            Velocities = backEnd.currentVelocities;
            Angulars = backEnd.currentAngulars;
            ReplayPoints.Clear();
        }

        protected static Vector3 FloatToVector(float[] input)
        {
            return new Vector3(input[0], input[1], input[2]);
        }

        protected void CalculateDeltas()
        {
            if (!isActiveAndEnabled) return;
            // if(Vector3.Distance(FloatToVector(explorePoints[replayPoints.Count - 1]), FloatToVector(replayPoints[replayPoints.Count-1])) > 0.1)
            //print("Significant Gap!");
            RigidBody.MovePosition(FloatToVector(ExplorePoints[ReplayPoints.Count - 1]));
            transform.eulerAngles = FloatToVector(Rotations[ReplayPoints.Count - 1]); 
            RigidBody.velocity = FloatToVector(Velocities[ReplayPoints.Count - 1]);
            RigidBody.angularVelocity = FloatToVector(Angulars[ReplayPoints.Count - 1]);
        }

        public void UpdateUI(string archiveSize, string iterations, string trajectories, string currentScore, string bestScore, string bestLength, string bestGame)
        {
            try
            {
                this.archiveSize.text = archiveSize;
                this.iterations.text = iterations;
                trajectorySize.text = trajectories;
                trajectoryScore.text = currentScore;
                bestScoreUI.text = bestScore;
                bestSizeUI.text = bestLength;
                bestGameScoreUI.text = bestGame;
            } catch (NullReferenceException) {}
        }
    
        public Vector3 GetPosition()
        {
            return transform.position;
        }

        public Vector3 GetRotation()
        {
            return transform.eulerAngles;
        }

        public Vector3 GetVelocity()
        {
            return RigidBody.velocity;
        }

        public Vector3 GetAngular()
        {
            return RigidBody.angularVelocity;
        }

        public abstract string[][] GetState();

        public abstract float BehaviorReward();

        public abstract float[][] GetInputs(bool expert);

        public abstract float GetScore();

        public abstract bool GetFinal();

    }
}
