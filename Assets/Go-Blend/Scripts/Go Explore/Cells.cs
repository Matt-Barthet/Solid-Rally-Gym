using System;
using System.Collections.Generic;
using System.Linq;
using Go_Blend.Scripts.Utilities;
using UnityEngine;

[Serializable]
public class Cell
{
    private readonly string[] state;
    public string[] Properties;
    private float reward, cumulativeReward;
    private List<int> trajectory;
    private List<float[]> actionTrajectory;
    private List<float> scoreTrajectory, arousalTrajectory;
    private List<string[]> stateTrajectory, propertiesTrajectory;
    private bool finalState;
    public float CumulativeArousal, CumulativeBehavior;
    public int Age, Visited;

    public List<float[]> coordinates, rotations, velocities, angulars;

    public List<float[]> surrogateTrace;
    public List<List<float>> arousalNieghbors;

    public Cell(IReadOnlyList<string[]> state, IEnumerable<int> trajectory, IEnumerable<float[]> actionTrajectory,
        IEnumerable<float> scoreTrajectory, IEnumerable<float> arousalTrajectory, IEnumerable<string[]> stateTrajectory, IEnumerable<string[]> propertiesTrajectory, IEnumerable<float[]> surrogateTrace, List<List<float>> arousalNeighbors)
    {
        this.state = state[0];
        Properties = state[1];
        this.trajectory = trajectory.ToList();
        this.actionTrajectory = actionTrajectory.ToList();
        this.scoreTrajectory = scoreTrajectory.ToList();
        this.arousalTrajectory = arousalTrajectory.ToList();
        this.stateTrajectory = stateTrajectory.ToList();
        this.propertiesTrajectory = propertiesTrajectory.ToList();
        this.surrogateTrace = surrogateTrace.ToList();

        try
        {
            this.arousalNieghbors = arousalNeighbors.ToList();
        }
        catch (ArgumentNullException)
        {
            Debug.LogError("Caught null neighbors");
            arousalNeighbors = new List<List<float>>();
        }

        finalState = false;
        Age = 0;
        Visited = 1;
        coordinates = new List<float[]>();
        rotations = new List<float[]>();
        angulars = new List<float[]>();
        velocities = new List<float[]>();
    }

    public void UpdateTrajectory(IEnumerable<int> pTrajectory, IEnumerable<float[]> pActionTrajectory,
        IEnumerable<float> pScoreTrajectory, IEnumerable<float> pArousalTrajectory, IEnumerable<string[]> pStateTrajectory, IEnumerable<string[]> pPropertiesTrajectory, IEnumerable<float[]> surrogateTrace, List<List<float>> arousalNeighbors)
    {
        trajectory = pTrajectory.ToList();
        actionTrajectory = pActionTrajectory.ToList();
        scoreTrajectory = pScoreTrajectory.ToList();
        arousalTrajectory = pArousalTrajectory.ToList();
        stateTrajectory = pStateTrajectory.ToList();
        propertiesTrajectory = pPropertiesTrajectory.ToList();
        this.surrogateTrace = surrogateTrace.ToList();
        this.arousalNieghbors = arousalNeighbors.ToList();
    }

    public List<int> GetTrajectory()
    {
        return trajectory;
    }

    public List<float[]> GetActionTrajectory()
    {
        return actionTrajectory;
    }

    public List<float> GetScoreTrace()
    {
        return scoreTrajectory;
    }

    public List<float> GetArousalTrace()
    {
        return arousalTrajectory;
    }

    public List<string[]> GetStateTrace()
    {
        return stateTrajectory;
    }    
    
    public List<string[]> GetPropertiesTrace()
    {
        return propertiesTrajectory;
    }

    public void UpdateReward(float pReward, float pCumulativeReward)
    {
        reward = pReward;
        cumulativeReward = pCumulativeReward;
    }

    public float GetReward()
    {
        return reward;
    }

    public float GetCumulativeReward()
    {
        return cumulativeReward;
    }

    public float GetArousalReward()
    {
        return CumulativeArousal / trajectory.Count;
    }

    public void SetEnvironment(string key)
    {
        StateSaveLoad.SaveEnvironment(key);
    }

    public void LoadEnvironment()
    {
        StateSaveLoad.LoadEnvironment(GetHashFromArray(state).ToString());
    }

    public void SetFinal(bool flag)
    {
        finalState = flag;
    }

    public bool GetFinal()
    {
        return finalState;
    }

    public void IncrementVisited()
    {
        Visited++;
    }

    public static int GetHashFromArray(IReadOnlyCollection<string> array)
    {
        unchecked
        {
            int hash = array.Count;
            foreach (string element in array)
            {
                hash = hash * 31 + element.GetHashCode();
            }
            return hash;
        }
    }
}
