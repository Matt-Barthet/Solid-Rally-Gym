using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using Go_Blend.Scripts.Car_Controllers;
using Go_Blend.Scripts.Car_Controllers.Base_Classes;
using Go_Blend.Scripts.Utilities;
using Pagan_Experiment;
using UnityEngine;

[CreateAssetMenu(menuName = "My Assets/Go Explore")]
public class GoExplore : ScriptableObject
{
    public static GoBlendBaseController PlayerController;
    public static DataLogger _dataLogger;

    public Dictionary<int, Cell> cellArchive;
    
    [HideInInspector]
    public Cell CurrentCell;
    private static readonly int[] Seeds = { 999, 10, 20, 30, 40, 50, 60, 70, 80, 90 };
    private const string Header = "Key,[general]Visited,Lap#,Speed,Track-Segment,Sub-Segment,AI Nearby,Facing-Target,Time_Passed,Position,Rotation,Velocity,Angular,[output]Steering,[output]Pedals,[general]Frames,[output]Arousal,Score";
    private Dictionary<string, List<float>> statistics;
    
    [HideInInspector]
    public List<int> currentTrajectory, bestTrajectory;
    public List<float[]> currentActionTrajectory, bestActionTrajectory;
    
    [HideInInspector]
    public List<float> currentScoreTrace, currentArousalTrace;
    public List<string[]> currentStateTrace, currentPropertiesTrace;

    public List<float[]> currentSurrogateTrace;

    public List<float []> currentCoordinates, currentRotations, currentVelocities, currentAngulars;
    
    [HideInInspector]
    public float[] surrogateVector = new float[32], closestPair = { 0, 0 };
    
    public List<float[]> ReplayActions;

    private int currentKey, currentStep, trajectoryCount;
    
    [HideInInspector]
    public bool exploring, savedState;
    
    [HideInInspector]
    public bool returning;

    private float cumulativeScore, cumulativeBehavior, cumulativeArousal;
    private float bestScore, bestArousal, bestBehavior, bestGameScore;
    private float currentScore, previousScore;
    public static float maxTrajectorySize;
    
    public int maxTrajectories = 500000;
    public int randomActionCount = 20;
    public int KNN = 5;
    public float lambda;
    public float epsilon = 0.01f;
    public string behaviorTarget = "Human";
    public string arousalTarget = "Human";
    public string arousalReward = "Predict_Arousal";
    public string selectionMethod = "Random";
    public int targetClusterNumber = 2;
    public int currentRun;
    public bool noReward;
    
    public List<List<float>> arousalNeigbors;

    public void Start()
    {
        _dataLogger = GameObject.FindGameObjectWithTag("DataLogger")?.GetComponent<DataLogger>();
        ProcessArguments();

        // HumanModel.Init(targetClusterNumber);
        UnityEngine.Random.InitState(Seeds[currentRun]);

        if (PlayerController.GetComponent<GoBlendCarController>() == null) return;
        
        // maxTrajectorySize = OutputAccuracy.human_arousal_trace.Count;
        
        returning = false;
        ReplayActions = new List<float[]>();
        currentTrajectory = new List<int>();
        currentActionTrajectory = new List<float[]>();
        currentScoreTrace = new List<float>();
        currentArousalTrace = new List<float>();
        currentStateTrace = new List<string[]>();
        currentPropertiesTrace = new List<string[]>();
        bestTrajectory = new List<int>();
        bestActionTrajectory = new List<float[]>();
        currentSurrogateTrace = new List<float[]>();
        arousalNeigbors = new List<List<float>>();

        statistics = new Dictionary<string, List<float>>
        {
            { "Archive Size", new List<float>{0} },
            { "Archive Age", new List<float>{0} },
            { "Mean Cell Length", new List<float>{0} },
            { "Best Cell Length", new List<float>{0} },
            { "Mean R_b", new List<float>{0} },
            { "Mean R_a", new List<float>{0} },
            { "Mean R_lambda", new List<float>{0} },
            { "Best R_b", new List<float>{0} },
            { "Best R_a", new List<float>{0} },
            { "Best R_lambda", new List<float>{0} },
            { "Humans Drawn", new List<float>{0} },
            { "Final Cells", new List<float>{0} }
        };

        currentCoordinates = new List<float[]>();
        currentAngulars = new List<float[]>();
        currentRotations = new List<float[]>();
        currentVelocities = new List<float[]>();
        currentStep = 0;
        trajectoryCount = 1;
        cumulativeScore = 0;
        cumulativeBehavior = 0;
        cumulativeArousal = 0;
        bestScore = 0;
        currentScore = 0;
        previousScore = 0;
        currentKey = Cell.GetHashFromArray(PlayerController.GetState()[0]);
        cellArchive = CellToCsv.LoadArchiveFromFile();

        if (cellArchive.Count > 0)
        {
            foreach (var statistic in statistics.Values)
                statistic.Clear();
        }
        else
        {
            using var resultsFile = new StreamWriter("./Data/Results_" + currentRun + ".csv");
            var headers = "";
            foreach (var statistic in statistics.Keys) headers += statistic + ",";
            resultsFile.WriteLine(headers.Remove(headers.Length - 1));
            resultsFile.Close();
        }
    }
    
    private void ProcessArguments()
    {
        var editor = false;
        #if UNITY_EDITOR
        editor = true;
        #endif
        
        PlayerController.GetInputs(true);
        
        if (editor) return;
        
        lambda = 1f;
        KNN = 5;
        targetClusterNumber = 2;
        randomActionCount = 20;
        epsilon = 0;
        noReward = false;
        behaviorTarget = "Imitate";
        arousalReward = "Predict_Arousal";
        currentRun = 0;
        selectionMethod = "Random";
        var args = Environment.GetCommandLineArgs();
        
        for (var i = 0; i < args.Length; i++)
        {
            switch (args[i])
            {
                case "-runNumber":
                    currentRun = int.Parse(args[i + 1]);
                    break;
                case "-lambdaValue":
                    lambda = float.Parse(args[i + 1]);
                    break;
                case "-exploreSteps":
                    randomActionCount = int.Parse(args[i + 1]);
                    break;
                case "-noReward" when bool.Parse(args[i + 1]):
                    noReward = true;
                    break;
                case "-epsilonValue":
                    epsilon = float.Parse(args[i + 1]);
                    break;
                case "-arousalTarget":
                    arousalTarget = args[i + 1];
                    break;
                case "-behaviorTarget":
                    behaviorTarget = args[i + 1];
                    break;
                case "-clusterNumber":
                    targetClusterNumber = int.Parse(args[i + 1]);
                    break;
                case "-arousalReward":
                    arousalReward = args[i + 1];
                    Debug.Log(arousalReward);
                    break;
                case "-selectionMethod":
                    selectionMethod = args[i + 1];
                    break;
            }
        }
        exploring = true;
    }

    public static float[] VectorToFloat(Vector3 input)
    {
        return new float[] { input.x, input.y, input.z };
    }

    public void ConstructCell(string[][] currentState, float[] pActionTaken)
    {
        CurrentCell = new Cell(currentState, currentTrajectory, currentActionTrajectory, currentScoreTrace,
            currentArousalTrace, currentStateTrace, currentPropertiesTrace, currentSurrogateTrace, arousalNeigbors);

        currentCoordinates.Add(VectorToFloat(PlayerController.GetPosition()));
        currentRotations.Add(VectorToFloat(PlayerController.GetRotation()));
        currentVelocities.Add(VectorToFloat(PlayerController.GetVelocity()));
        currentAngulars.Add(VectorToFloat(PlayerController.GetAngular()));

        currentKey = Cell.GetHashFromArray(currentState[0]);
        currentTrajectory.Add(currentKey);
        currentActionTrajectory.Add(pActionTaken);
        currentStateTrace.Add(currentState[0]);
        currentPropertiesTrace.Add(currentState[1]);

        foreach (var cell in cellArchive.Values)
            cell.Age++;
    }

    public void AssessCell()
    {
        var arousal = 0f;
        var stDev = 0f;
        surrogateVector = _dataLogger?.GetSurrogateVector() ?? new float[32];

        closestPair = HumanModel.GetClosestHuman(surrogateVector);
        stDev = closestPair[0];
        arousal = closestPair[1];

        currentScoreTrace.Add(PlayerController.GetScore());
        currentArousalTrace.Add(arousal);
        // arousalNeigbors.Add(OutputAccuracy.arousalSample);
        currentSurrogateTrace.Add(surrogateVector);

        CurrentCell.UpdateReward(ComputeReward(arousal, stDev), cumulativeScore);
        CurrentCell.UpdateTrajectory(currentTrajectory, currentActionTrajectory, currentScoreTrace, currentArousalTrace, currentStateTrace, currentPropertiesTrace, currentSurrogateTrace, arousalNeigbors);

        var archiveSize = cellArchive.Count.ToString();
        var trajectories = trajectoryCount.ToString();
        var currentLength = currentTrajectory.Count.ToString();
        var currentReward = cumulativeBehavior * (1 - lambda) + " + " + cumulativeArousal * lambda / currentTrajectory.Count;
        var bestReward = bestScore.ToString(CultureInfo.InvariantCulture);
        var bestLength = bestActionTrajectory.Count.ToString();
        var bestGame = bestGameScore.ToString(CultureInfo.InvariantCulture);
        PlayerController.UpdateUI(archiveSize, trajectories, currentLength, currentReward, bestReward, bestLength, bestGame);

        CurrentCell.SetFinal(currentTrajectory.Count >= maxTrajectorySize || PlayerController.GetFinal());
        StoreCellProcedure(currentKey, CurrentCell);

        if (++currentStep < randomActionCount && !CurrentCell.GetFinal()) return;
        if (trajectoryCount % 10000 == 0 & trajectoryCount > 0) SaveBestTrajectories();

        StartNewTrajectory();
        trajectoryCount++;

        if (trajectoryCount >= maxTrajectories)
        {
            Application.Quit();
            exploring = false;
            Time.timeScale = 0f;
        }
    }

    private int ChooseCell()
    {
        // Debug.Log("Selecting next cell randomly...");
        while (true)
        {
            var index = UnityEngine.Random.Range(0, cellArchive.Count);
            var counter = 0;
            foreach (var pair in cellArchive.Where(pair => counter++ == index && !pair.Value.GetFinal()))
            {
                return pair.Key;
            }
        }
    }

    private int ChooseCellArousal()
    {
        Debug.Log("Selecting next cell via roulette wheel (arousal)...");
        while (true)
        {
            var weightSum = cellArchive.Where(cell => !cell.Value.GetFinal()).Sum(cell => cell.Value.GetArousalReward());
            var index = UnityEngine.Random.Range(0, weightSum);
            foreach (var cell in cellArchive)
            {
                if (!cell.Value.GetFinal() && index <= cell.Value.GetArousalReward())
                    return cell.Key;
                index -= cell.Value.GetArousalReward();
            }
            Debug.LogError("WRF Error!");
            return 0;
        }
    }

    private void StoreCellProcedure(int key, Cell temp)
    {
        if (currentTrajectory.Count > maxTrajectorySize) return;
        if (cumulativeScore < 0) return;
        if (!PlayerController.legalState) return;

        if (!cellArchive.ContainsKey(key))
        {
            temp.coordinates = new List<float[]>(currentCoordinates);
            temp.rotations = new List<float[]>(currentRotations);
            temp.velocities = new List<float[]>(currentVelocities);
            temp.angulars = new List<float[]>(currentAngulars);
            temp.SetEnvironment(key.ToString());
            cellArchive.Add(key, temp);
        }
        else
        {
            if (!ShouldStoreCell(key, temp)) return;
            cellArchive[key].UpdateReward(temp.GetReward(), temp.GetCumulativeReward());
            cellArchive[key].CumulativeArousal = cumulativeArousal;
            cellArchive[key].CumulativeBehavior = cumulativeBehavior;
            cellArchive[key].UpdateTrajectory(currentTrajectory, currentActionTrajectory, currentScoreTrace, currentArousalTrace, currentStateTrace, currentPropertiesTrace, currentSurrogateTrace, arousalNeigbors);
            cellArchive[key].SetEnvironment(key.ToString());
            cellArchive[key].Properties = temp.Properties;
            cellArchive[key].Age = 0;
            cellArchive[key].coordinates = new List<float[]>(currentCoordinates);
            cellArchive[key].rotations = new List<float[]>(currentRotations);
            cellArchive[key].velocities = new List<float[]>(currentVelocities);
            cellArchive[key].angulars = new List<float[]>(currentAngulars);
        }
    }

    private bool ShouldStoreCell(int key, Cell cell)
    {
        if (!cellArchive.ContainsKey(key))
            return true;
        if (cell.GetCumulativeReward() < cellArchive[key].GetCumulativeReward())
            return false;
        if (cell.GetCumulativeReward() >= cellArchive[key].GetCumulativeReward() + epsilon)
            return true;
        return cell.GetTrajectory().Count < cellArchive[key].GetTrajectory().Count;
    }

    private float ComputeReward(float arousal, float stDev)
    {
        /*var score = 0f;
        float target_arousal = 1;

        if (behaviorTarget == "Optimal")
        {
            score = OptimalBehaviorReward();
        } 
        else if (behaviorTarget == "Imitate")
        {
            try
            {
                score = ScoreImitation(PlayerController.GetScore(), (float)OutputAccuracy.human_score_trace[currentTrajectory.Count - 1]);
            } catch (Exception)
            {
                score = ScoreImitation(PlayerController.GetScore(), (float)OutputAccuracy.human_score_trace[currentTrajectory.Count - 2]);
            }
        }

        try
        {
            if (arousalTarget == "Human") target_arousal = (float)OutputAccuracy.human_arousal_trace[currentTrajectory.Count - 1];
            else if (arousalTarget == "Minimize") target_arousal = 0;
            else if (arousalTarget == "Maximize") target_arousal = 1;
            if(arousalReward == "Classify_Arousal_Value") stDev = (float)OutputAccuracy.human_arousal_variance[currentTrajectory.Count - 1];
        } catch (Exception)
        {
            if (arousalTarget == "Human") target_arousal = (float)OutputAccuracy.human_arousal_trace[currentTrajectory.Count - 2];
            if (arousalReward == "Classify_Arousal_Value") stDev = (float)OutputAccuracy.human_arousal_variance[currentTrajectory.Count - 2];
        }


        // If we are trying to predict a value within human confidence range:
        if (arousalReward == "Classify_Arousal_Value" && arousalTarget == "Human")
        {
            score += ClassifyAffect(arousal, target_arousal, stDev);
        }
        // If we are trying to constantly increase arousal from one window to the next.
        else if (arousalReward == "Classify_Arousal_Value" && arousalTarget == "Maximize")
        {
            try
            {
                score += IncreaseArousal(arousal, currentArousalTrace[currentArousalTrace.Count - 2]);
            }
            catch (ArgumentOutOfRangeException)
            {
                Debug.LogWarning("First window being evaluated...");
                score += IncreaseArousal(arousal, 0);
            }
        }
        // If we are trying to predict an increase, decrease or stable arousal change in humans
        else if (arousalReward == "Classify_Arousal_Delta")
        {
            try
            {
                score += ClassifyAffectDelta(arousal - currentArousalTrace[currentArousalTrace.Count - 2], (float)OutputAccuracy.human_arousal_trace[currentTrajectory.Count - 1] - (float)OutputAccuracy.human_arousal_trace[currentTrajectory.Count - 2]);
            }
            catch (ArgumentOutOfRangeException)
            {
                Debug.LogWarning("First window being evaluated...");
                score += ClassifyAffectDelta(arousal, (float)OutputAccuracy.human_arousal_trace[currentTrajectory.Count - 1]);
            }
        }
        // If we are trying to predict the absolute change in arousal from last time windows...
        else if (arousalReward == "Predict_Arousal_Delta")
        {
            try
            {
                score += OptimalAffectReward(arousal - currentArousalTrace[currentArousalTrace.Count - 2], (float)OutputAccuracy.human_arousal_trace[currentTrajectory.Count - 1] - (float)OutputAccuracy.human_arousal_trace[currentTrajectory.Count - 2], 0);
            }
            catch (ArgumentOutOfRangeException)
            {
                Debug.LogWarning("First window being evaluated...");
                score += OptimalAffectReward(arousal, (float)OutputAccuracy.human_arousal_trace[currentTrajectory.Count - 1], 0);
            }
        }
        // Otherwise use the standard reward.
        else if (arousalReward == "Predict_Arousal" || arousalReward == "Predict_Arousal_Stdev")
        {
            score += OptimalAffectReward(arousal, target_arousal, stDev);
        } else
        {
            throw new Exception("No arousal function provided.");
        }

        if (noReward)
        {
            score = 0;
            if (cumulativeArousal > bestArousal)
                bestArousal = cumulativeArousal;
            if (cumulativeBehavior > bestBehavior)
                bestBehavior = cumulativeBehavior;
            if (PlayerController.GetScore() > bestGameScore)
                bestGameScore = PlayerController.GetScore();
        }
        else
        {

            // Arousal Component
            cumulativeScore = cumulativeArousal * lambda;
            if (arousalReward == "Predict_Arousal" || arousalReward == "Predict_Arousal_Stdev" || arousalReward == "Predict_Arousal_Delta")
            {
                cumulativeScore /= currentTrajectory.Count;
            }
            else
            {
                cumulativeScore /= OutputAccuracy.human_arousal_trace.Count;
            }

            // Behavior Component
            if (behaviorTarget == "Imitate")
                cumulativeScore += cumulativeBehavior * (1 - lambda) / currentTrajectory.Count;
            else
                cumulativeScore += cumulativeBehavior * (1 - lambda);


            // Updating std output if a new cumulative best score is found.
            if (cumulativeScore <= bestScore) return score;
            bestScore = cumulativeScore;
            bestTrajectory = currentTrajectory.ToList();
            bestActionTrajectory = currentActionTrajectory.ToList();
            bestGameScore = PlayerController.GetScore();

            // Normalize arousal based on the reward function used.
            bestArousal = cumulativeArousal;
            if (arousalReward == "Predict_Arousal" || arousalReward == "Predict_Arousal_Stdev" || arousalReward == "Predict_Arousal_Delta")  bestArousal /= currentTrajectory.Count;
            else bestArousal /= OutputAccuracy.human_arousal_trace.Count;

            // Normalize behavior reward if we are imitating humans.
            bestBehavior = cumulativeBehavior;
            if (behaviorTarget == "Imitate") bestBehavior /= currentTrajectory.Count;
        }

        return score;*/
        return 0;
    }

    private float ScoreImitation(float score, float target)
    {
        var scoreDelta =  (float) Math.Pow(1 - Math.Abs(target - score) / 16, 2);
        cumulativeBehavior += scoreDelta;
        CurrentCell.CumulativeBehavior = cumulativeBehavior;
        return scoreDelta * (1 - lambda);
    }

    private float OptimalBehaviorReward()
    {
        currentScore = PlayerController.BehaviorReward();
        previousScore = cumulativeBehavior;
        var behaviorDelta = currentScore - previousScore;
        cumulativeBehavior += behaviorDelta;
        CurrentCell.CumulativeBehavior = cumulativeBehavior;
        return behaviorDelta * (1-lambda);
    }

    private float OptimalAffectReward(float arousal, float target, float stDev)
    {
        var arousalDelta = 0f;
        if (arousalReward == "Predict_Arousal_Delta")
        {
            arousalDelta = (float)Math.Pow(1 - Utility.MinMaxScaling(Math.Abs(target - arousal), 0, 2), 2);
            // Debug.Log($"{arousal} from {target} = {arousalDelta}");
        } else
        {
            arousalDelta = (float)Math.Pow(1 - Math.Abs(target - arousal), 2);
            if (arousalReward == "Predict_Arousal_Stdev")
            {
                // Debug.Log($"{arousalDelta} / {stDev + 1}");
                arousalDelta /= (float)(stDev + 1);
            }
        }

        cumulativeArousal += arousalDelta;
        CurrentCell.CumulativeArousal = cumulativeArousal;
        return arousalDelta * lambda;
    }

    private float ClassifyAffect(float arousal, float target, float targetDev)
    {
        if (arousal >= target - targetDev && arousal <= target + targetDev)
        {
            // Debug.Log($"{arousal}: [{target - targetDev} <-> {target + targetDev}]");
            cumulativeArousal += 1;
            CurrentCell.CumulativeArousal = cumulativeArousal;
            return 1 * lambda;
        } 
        else
        {
            cumulativeArousal -= 1;
            CurrentCell.CumulativeArousal = cumulativeArousal;
            return -1 * lambda;
        }

    }

    private float IncreaseArousal(float arousal, float previous)
    {
        if (arousal > previous)
        {
            // Debug.Log($"{arousal} > {previous}");
            cumulativeArousal += 1;
            CurrentCell.CumulativeArousal = cumulativeArousal;
            return 1 * lambda;
        } else
        {
            cumulativeArousal -= 1;
            CurrentCell.CumulativeArousal = cumulativeArousal;
            return -1 * lambda;
        }
    }

    private float ClassifyAffectDelta(float agent_delta, float human_delta)
    {
        if ((agent_delta > 0 && human_delta > 0) || (agent_delta < 0 && human_delta < 0) || (agent_delta == human_delta))
        {
            // Debug.Log($"{agent_delta} ||| {human_delta}");
            cumulativeArousal += 1;
            CurrentCell.CumulativeArousal = cumulativeArousal;
            return 1 * lambda;
        } else
        {
            // Debug.Log($"{agent_delta} =/= {human_delta}");
            cumulativeArousal -= 1;
            CurrentCell.CumulativeArousal = cumulativeArousal;
            return -1 * lambda;
        }
    }

    private void UpdateStats()
    {
        float rA = 0, rB = 0, rLambda = 0, age = 0, length = 0, final = 0;
        foreach (var cell in cellArchive.Values)
        {
            if (cell.GetTrajectory().Count > 0) rA += cell.CumulativeArousal / cell.GetTrajectory().Count;
            rB += cell.CumulativeBehavior;
            if (behaviorTarget == "Imitate" && cell.GetTrajectory().Count > 0) rB /= cell.GetTrajectory().Count;
            rLambda += cell.GetCumulativeReward();
            age += cell.Age;
            length += cell.GetActionTrajectory().Count;
            if (cell.GetFinal()) final++;
        }

        statistics["Archive Size"].Add(cellArchive.Count);
        statistics["Mean Cell Length"].Add(length / cellArchive.Count);
        statistics["Best Cell Length"].Add(bestTrajectory.Count);
        statistics["Archive Age"].Add(age / cellArchive.Count);
        statistics["Mean R_a"].Add(rA / cellArchive.Count);
        statistics["Mean R_b"].Add(rB / cellArchive.Count);
        statistics["Mean R_lambda"].Add(rLambda / cellArchive.Count);
        statistics["Best R_a"].Add(bestArousal / bestTrajectory.Count);
        statistics["Best R_b"].Add(bestBehavior);
        statistics["Best R_lambda"].Add(bestScore);
        // statistics["Humans Drawn"].Add(OutputAccuracy._HumansDrawn.Count);
        statistics["Final Cells"].Add(final);
    }

    public void StartNewTrajectory()
    {
        currentStep = 0;

        if (selectionMethod == "Random")
        {
            currentKey = ChooseCell();
        } else if (selectionMethod == "Arousal")
        {
            currentKey = ChooseCellArousal();
        } else
        {
            throw new Exception("No selection method provided.");
        }

        CurrentCell = cellArchive[currentKey];
        previousScore = cellArchive[currentKey].CumulativeBehavior;

        cumulativeScore = cellArchive[currentKey].GetCumulativeReward();
        cumulativeArousal = cellArchive[currentKey].CumulativeArousal;
        cumulativeBehavior = cellArchive[currentKey].CumulativeBehavior;

        currentTrajectory = cellArchive[currentKey].GetTrajectory().ToList();
        currentActionTrajectory = cellArchive[currentKey].GetActionTrajectory().ToList();
        currentArousalTrace = cellArchive[currentKey].GetArousalTrace().ToList();
        currentScoreTrace = cellArchive[currentKey].GetScoreTrace().ToList();
        currentStateTrace = cellArchive[currentKey].GetStateTrace().ToList();
        currentPropertiesTrace = cellArchive[currentKey].GetPropertiesTrace().ToList();
        currentSurrogateTrace = cellArchive[currentKey].surrogateTrace.ToList();
        arousalNeigbors = cellArchive[currentKey].arousalNieghbors.ToList();

        cellArchive[currentKey].IncrementVisited();

        currentCoordinates = new List<float[]>(cellArchive[currentKey].coordinates);
        currentRotations = new List<float[]>(cellArchive[currentKey].rotations);
        currentVelocities = new List<float[]>(cellArchive[currentKey].velocities);
        currentAngulars = new List<float[]>(cellArchive[currentKey].angulars);
        Debug.Log($"Current Iteration: {trajectoryCount}, Best Score: {bestScore}, Best Size: {bestTrajectory.Count}, " +
            $"Best Game Score: {bestGameScore}, Cell count: {cellArchive.Count}");

        UpdateStats();
        ReplayActions = CurrentCell.GetActionTrajectory();
        returning = true;
    }

    private int ChooseBestAffect(bool sizeInvariant)
    {
        var bestKey = 0;
        Cell best = null;
        float bestReward = -10000;
        foreach (var cell in cellArchive.Where(cell => cell.Value.CumulativeArousal > bestReward && (cell.Value.GetTrajectory().Count >= maxTrajectorySize || sizeInvariant)))
        {
            bestReward = cell.Value.CumulativeArousal;
            best = cell.Value;
            bestKey = cell.Key;
        }
        return bestKey;
    }

    private int ChooseBestBehavior(bool sizeInvariant)
    {
        var bestKey = 0;
        Cell best = null;
        float bestReward = -10000;
        foreach (var cell in cellArchive.Where(cell => cell.Value.CumulativeBehavior > bestReward && (cell.Value.GetTrajectory().Count >= maxTrajectorySize || sizeInvariant)))
        {
            bestReward = cell.Value.CumulativeBehavior;
            best = cell.Value;
            bestKey = cell.Key;
        }
        return bestKey;
    }

    private int ChooseBestCell(bool sizeInvariant)
    {
        var bestKey = 0;
        Cell best = null;
        float bestReward = -10000;
        foreach (var cell in cellArchive.Where(cell => cell.Value.GetCumulativeReward() > bestReward && (cell.Value.GetTrajectory().Count >= maxTrajectorySize || sizeInvariant)))
        {
            bestReward = cell.Value.GetCumulativeReward();
            best = cell.Value;
            bestKey = cell.Key;
        }
        return bestKey;
    }

    public void SaveCell(int key, string name)
    {
        if (key == 0) return;
        var best = cellArchive[key];
        SaveCell(best, $"./Data/Best {name} Cell.csv");
    }

    public static void SaveCell(Cell cell, string path)
    {
        TextWriter tw = new StreamWriter($"{path}.csv");
        tw.WriteLine(Header);
        for (var j = 0; j < cell.GetActionTrajectory().Count; j++)
            SaveCellState(tw, cell, j);
        tw.Close();
    }

    public void SaveBestCells()
    {
        if (!noReward)
        {
            SaveCell(ChooseBestCell(false), "of Full Length Lambda");
            SaveCell(ChooseBestCell(true), "of Any Length Lambda");
        }
        SaveCell(ChooseBestBehavior(true), "of Any Length Behavior");
        SaveCell(ChooseBestAffect(false), "of Full Length Affect");
    }

    public static void SaveCellState(TextWriter tw, Cell cell, int timeWindow)
    {
        var state = cell.GetStateTrace()[timeWindow];
        var line = cell.GetTrajectory()[timeWindow] + "," + cell.Visited + "," + state[4] + "," + state[0] + "," + state[1] + "," + state[2] + "," + state[3] + "," + state[5];
        line += "," + timeWindow * 0.25 + "," + cell.GetPropertiesTrace()[timeWindow][0] + "," + cell.GetPropertiesTrace()[timeWindow][1] + "," + cell.GetPropertiesTrace()[timeWindow][2] + "," + cell.GetPropertiesTrace()[timeWindow][3];
        line += "," + cell.GetActionTrajectory()[timeWindow][0] + "," + cell.GetActionTrajectory()[timeWindow][1] + "," + cell.GetActionTrajectory()[timeWindow][2] + "," + cell.GetArousalTrace()[timeWindow] + "," + cell.GetScoreTrace()[timeWindow];
        tw.WriteLine(line);
    }

    private void SaveBestTrajectories()
    {
        using (var resultsFile = File.AppendText("./Data/Results_" + currentRun + ".csv"))
        {
            for (var i = 0; i < statistics["Mean R_a"].Count; i++)
            {
                var line = "";
                foreach (var statistic in statistics.Values)
                    line += statistic[i] + ",";
                resultsFile.WriteLine(line.Remove(line.Length - 1));
            }
            resultsFile.Close();
        }

        foreach (var statistic in statistics.Values)
            statistic.Clear();

        var tw = new StreamWriter("./Data/Archive_" + currentRun + ".csv");
        tw.WriteLine(Header);
        foreach (var cell in cellArchive)
        {
            var size = cell.Value.GetTrajectory().Count - 1;
            if (size < 0)
            {
                continue;
            }
            SaveCellState(tw, cell.Value, size);
        }
        tw.Close();
        SaveBestCells();
    }
    
}

