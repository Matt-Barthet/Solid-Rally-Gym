using System.Collections.Generic;
using UnityEngine;
using System;
using MoreMountains.HighroadEngine;
using System.IO;
using System.Linq;
using Go_Blend.Scripts.Car_Controllers.Base_Classes;
using Go_Blend.Scripts.Utilities;
using Pagan_Experiment;
using UnityEngine.Serialization;

public class GoBlendReplayController : GoBlendSolidController
{
    [HideInInspector]
    public DataLogger logger;
    
    private List<float[]> replayList;
    private float lastTime;
    Vector3 startingPos, startingVel, startingAngular;
    Quaternion startingRot;
    
    [FormerlySerializedAs("arousal_values")] [HideInInspector]
    public List<float> arousalValues;

    [FormerlySerializedAs("score_values")] [HideInInspector]
    public List<float> scoreValues;

    public int run;
    public bool replayActions = true;
    public string cellName = "Best_Cell";
    public string experimentName = "Pro" ;
    public string agentName = "R0.5";
    
    private DirectoryInfo info;
    private FileInfo[] fileInfo; 
    List<float> scores, botDistances, speeds, offroads, midairs, crashings;
    private int counter;
    private string newStateHeaders = "Position_X,Position_Y,Position_Z,Velocity_X,Velocity_Y,Velocity_Z,Rotation_X,Rotation_Y,Rotation_Z,Raycast_1,Raycast_2,Raycast_3,Raycast_4,Raycast_5,Raycast_6,Raycast_7,Raycast_8,[output]Steering,[output]Pedals,[output]Arousal";
    private string filename;
    private List<float[]> originalReplay;
    private List<float> deviations;
    private List<float[]> newState;

    private List<float> steerings;
    private List<float> pedals;

    public bool showGizmos;

    protected new void Start()
    {
        base.Start();
        Time.captureFramerate = framerate;
        Time.fixedDeltaTime = 0.025f;
        RigidBody = GetComponent<Rigidbody>();
        SolidController = GetComponent<SolidController>();
        solidWheels = GetComponentsInChildren<SolidWheelBehaviour>();
        startingPos = transform.position;
        startingRot = transform.rotation;
        startingVel = Vector3.zero;
        startingAngular = Vector3.zero;
        // string path = $"./Evaluation/{experimentName}/{agentName}/Run{run}/Data/Cells";
        string path = "./Evaluation/";
        info = new DirectoryInfo(path);
        fileInfo = info.GetFiles();
        Initialize();
    }

    public new void Initialize()
    {
        if (!isActiveAndEnabled) return;

        base.Initialize();
        
        lastTime = 0;
        currentTime = 0.25f;
        ReplayPoints.Clear();

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

        deviations = new List<float>();
        scores = new List<float>();
        botDistances = new List<float>();
        speeds = new List<float>();
        midairs = new List<float>();
        offroads = new List<float>();
        crashings = new List<float>();
        replayList = new List<float[]>();

        logger = FindObjectOfType<DataLogger>();
        foreach (var wheel in solidWheels) wheel.Reset();
        Resources.UnloadUnusedAssets();

        replayList = new List<float[]>();
        ExplorePoints = new List<float[]>();
        Rotations = new List<float[]>();
        Velocities = new List<float[]>();
        Angulars = new List<float[]>();
        newState = new List<float[]>();
        arousalValues = new List<float>();
        scoreValues = new List<float>();

        steerings = new List<float>();
        pedals = new List<float>();
        
        if (keyboard) return;
        while (true)
        {
            var file = fileInfo[counter++];
            var split = file.ToString().Split('/');
            filename = split[split.Length - 1];
            filename = $"Cells/{filename}";
            if(filename.Contains(".csv") && !filename.Contains("New"))
            {
                LoadTrace(filename);
                break;
            }
        }
    }
    
    private void LoadTrace(string name)
    {
        print("Loading trace!");

        using var reader = new StreamReader(@$"./Evaluation/{experimentName}/{agentName}/Run{run}/Data/{name}");
        reader.ReadLine();
        while (!reader.EndOfStream)
        {
            var line = reader.ReadLine();
            var values = line.Split(',');
            replayList.Add(new float[] { float.Parse(values[13]), float.Parse(values[14]) });
            
            var coordinate = values[9].Split("|");
            ExplorePoints.Add(new float[] { float.Parse(coordinate[0]), float.Parse(coordinate[1]), float.Parse(coordinate[2]) });

            var rotation = values[10].Split("|");
            Rotations.Add(new float[] { float.Parse(rotation[0]), float.Parse(rotation[1]), float.Parse(rotation[2]) });

            var velocity = values[11].Split("|");
            Velocities.Add(new float[] { float.Parse(velocity[0]), float.Parse(velocity[1]), float.Parse(velocity[2]) });

            var angular = values[12].Split("|");
            Angulars.Add(new float[] { float.Parse(angular[0]), float.Parse(angular[1]), float.Parse(angular[2]) });

            var arousal = values[values.Length - 2];
            arousalValues.Add(float.Parse(arousal));

            var score = values[values.Length - 1];
            scoreValues.Add(float.Parse(score));
        }

        originalReplay = replayList.ToList();
    }

    private void OnDestroy()
    {
        if (!isActiveAndEnabled) return;
        using var resultsFile = new StreamWriter("./Deviations.csv");
        resultsFile.WriteLine("Deviation");
        foreach (var statistic in deviations)
            resultsFile.WriteLine(statistic);
        resultsFile.Close();
    }

    private void ReplayAction()
    {
        /*var surrogateVector = logger.GetSurrogateVector() ?? new float[32];
        var values = OutputAccuracy.GetClosestHuman(surrogateVector, 5);
        deviations.Add(values[0]);
        arousal_values.Add(values[1]);
        scores.Add(GetScore());
        botDistances.Add(surrogateVector[29]);
        speeds.Add(surrogateVector[2]);
        crashings.Add(surrogateVector[6]);
        offroads.Add(surrogateVector[7]);
        midairs.Add(surrogateVector[4]);*/

        var replay = replayList[0];
        ReplayPoints.Add(GoExplore.VectorToFloat(GetPosition()));
        replayList.RemoveAt(0);
        SolidController.VerticalPosition(replay[1]);
        SolidController.HorizontalPosition(replay[0]);
        CalculateDeltas();
        UpdateNewState();
    }
    
    private void UpdateNewState()
    {
        var precision = 1;

        float[] currentState = new float[17];
        
        // Position of the car rounded to the nearest precision point.
        var position = transform.position;
        position = new Vector3((float) Math.Ceiling(position.x / precision) * precision,
            (float) Math.Ceiling(position.y / precision) * precision,
            (float) Math.Ceiling(position.z / precision) * precision
            );

        currentState[0] = position.x;
        currentState[1] = position.y;
        currentState[2] = position.z;

        // Velocity of the car rounded to the nearest precision point.
        var velocity = RigidBody.velocity;
        velocity = new Vector3((float) Math.Ceiling(velocity.x / precision) * precision,
            (float) Math.Ceiling(velocity.y / precision) * precision,
            (float) Math.Ceiling(velocity.z / precision) * precision
        );

        currentState[3] = velocity.x;
        currentState[4] = velocity.y;
        currentState[5] = velocity.z;
        
        // Rotation of the car rounded to the nearest precision point.
        var rotation = transform.eulerAngles;
        rotation = new Vector3((float) Math.Ceiling(rotation.x / precision) * precision,
            (float) Math.Ceiling(rotation.y / precision) * precision,
            (float) Math.Ceiling(rotation.z / precision) * precision
        );

        currentState[6] = rotation.x;
        currentState[7] = rotation.y;
        currentState[8] = rotation.z;
        
        var startingVec = Vector3.forward;
        int numberOfRaycasts = 8, maxDistance = 100;    
        int layerMask = ~LayerMask.GetMask("Ignore Raycast" );

        for (int i = 0; i < numberOfRaycasts; i++)
        {            
            RaycastHit hit;
            var direction = transform.TransformDirection(Quaternion.Euler(0, -360 / numberOfRaycasts * i, 0) * startingVec);
            
            if (Physics.Raycast(transform.position, direction, out hit, maxDistance, layerMask))
            {
                Debug.DrawRay(transform.position, direction * hit.distance, Color.white);
                currentState[i + 9] = (float) Math.Ceiling(hit.distance / precision) * precision;
            }
            else
            {
                currentState[i + 9] = maxDistance;
            }
        }
        steerings.Add(SolidController.CurrentSteeringAmount);
        pedals.Add(SolidController.CurrentGasPedalAmount);
        newState.Add(currentState);
    }
    
    private bool keyboard = true;
    
    protected void FixedUpdate()
    {
        if (!replayActions) return;
        currentTime += Time.fixedDeltaTime;
        
        if (keyboard)
        {
            var targetSteering = Input.GetAxisRaw("Horizontal");
            var targetPedals = Input.GetAxisRaw("Vertical");
            SolidController.VerticalPosition(targetPedals);
            SolidController.HorizontalPosition(targetSteering);
            if (SolidController.CurrentLap >= 1)
            {
                keyboard = false;
            }
            UpdateNewState();
        }
        
        // if (Math.Round(currentTime - lastTime, 3) < 0.25f) return;
        
        print("Adding");
        if (replayList.Count > 0 && !keyboard)
        {
            ReplayAction();
        }
        else if (!keyboard)
        {
            // using (StreamWriter resultsFile = new StreamWriter($"./Evaluation/{experimentName}/{agentName}/Run{run}/Data/{filename.Insert(6, "NEW/")}"))
            using StreamWriter resultsFile = new StreamWriter($"./Evaluation/Human_Trace.csv");
            resultsFile.WriteLine(newStateHeaders);
            for(var j = 0; j < newState.Count; j++)
            {
                string newLine = "";
                for(var i = 0; i < newState[j].Length; i++)
                {
                    newLine += $"{newState[j][i]},";
                }
                newLine += $"{steerings[j]},{pedals[j]},{0}";
                resultsFile.WriteLine(newLine);
                /*resultsFile.WriteLine("Crashing,BotDistance,OffRoad,MidAir,Speed");
                    print(crashings.Count);
                    for (var i = 0; i < crashings.Count; i++)
                    {
                        resultsFile.WriteLine($"{crashings[i]},{botDistances[i]},{offroads[i]},{midairs[i]},{speeds[i]}");
                    }*/
            }
            resultsFile.Close();
            Destroy(gameObject);
            // restart = true;
            /*TextWriter tw = new StreamWriter($"{name}_arousal.csv");
                for (var j = 0; j <arousal_values.Count; j++)
                {
                    string line = $"{arousal_values[j]}\n";
                    tw.Write(line);
                }
                tw.Close();
                restart = true;
            */
        }
        lastTime = currentTime;
    }
    
    public void OnDrawGizmos()
    {
        if (!isActiveAndEnabled) return;
        if (!showGizmos) return;
#if UNITY_EDITOR
        if (Application.isPlaying)
        {

            var currentLap = SolidController.CurrentLap;

            for (int i = 0; i < ReplayPoints.Count; i++)
            {
                // Math.Pow(1 - Math.Abs(target - arousal), 2)
                // var arousal_delta = (float) Math.Pow(1 - Math.Abs(OutputAccuracy.human_arousal_trace[i] - arousal_values[i]), 2);
                // var arousal_delta = (float) OutputAccuracy.human_arousal_trace[i] - arousal_values[i];
                
                /*if (arousal_delta < 0)
                {
                    Gizmos.color = Color.Lerp(Color.blue, Color.white, Utility.MinMaxScaling(arousal_delta, -1f, 0));
                } else
                {
                    Gizmos.color = Color.Lerp(Color.white, Color.red, Utility.MinMaxScaling(arousal_delta, 0f, 1f));
                }*/

                if (arousalValues[i] < 0.5)
                {
                    Gizmos.color = Color.Lerp(Color.blue * 1.5f, Color.white, arousalValues[i] * 2);
                }
                else
                {
                    Gizmos.color = Color.Lerp(Color.white, Color.red * 1.5f, Utility.MinMaxScaling(arousalValues[i], 0.5f, 1));
                }

                if (scoreValues[i] < currentLap * 8 || scoreValues[i] > (currentLap + 1) * 8) continue;
                Gizmos.DrawSphere(FloatToVector(ExplorePoints[i]), 5f);
                if (i > 0) Gizmos.DrawLine(FloatToVector(ExplorePoints[i]), FloatToVector(ExplorePoints[i - 1]));
            }

            /*Gizmos.color = Color.blue;
            for (int i = 0; i < replayPoints.Count; i++)
            {
                Gizmos.DrawSphere(FloatToVector(replayPoints[i]), 0.5f);
                try
                {
                    Gizmos.DrawLine(FloatToVector(replayPoints[i]), FloatToVector(explorePoints[i]));
                    if (i > 0)
                        Gizmos.DrawLine(FloatToVector(replayPoints[i]), FloatToVector(replayPoints[i - 1]));
                }
                catch (ArgumentOutOfRangeException) { }
            }
            Gizmos.color = Color.black;
            for (int i = 0; i < replayPoints.Count; i++)
            {
                try
                {
                    Gizmos.DrawLine(FloatToVector(replayPoints[i]), FloatToVector(explorePoints[i]));
                    Handles.Label(FloatToVector(replayPoints[i]) + new Vector3(0, 2, 0), deltas[i].ToString());
                }
                catch (ArgumentOutOfRangeException) { }
            }*/
        }
#endif
    }
    
    private void LateUpdate()
    {
        if (Restart)
        {
            Restart = false;
            StateSaveLoad.Reset();
        }
    }
    
    public override float[][] GetInputs(bool expert)
    {
        return new[] { new float[10], new float[10] };
    }
    
    public override string[][] GetState()
    {
        var newState = new string[6] {"", "", "", "", "", ""};
        var properties = new string[4] {"", "", "", ""};
        return new[] { newState, properties };
    }
    
    public override float BehaviorReward()
    {
        return 0;
    }

    public override bool GetFinal()
    {
        return SolidController.CurrentLap >= 3;
    }

    public override float GetScore()
    {
        return SolidController.Score;
    }

}
