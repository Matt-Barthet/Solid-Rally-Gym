using System;
using System.Collections.Generic;
using System.Linq;

namespace Go_Blend.Scripts.Utilities
{
    public static class Utility
    {
        public static float MinMaxScaling(float value, float min, float max)
        {
            var numerator = value - min;
            var denominator = max - min;
            if (denominator == 0)
                return value;
            return numerator / denominator;
        }

        public static double Distance(IEnumerable<float> array1, IReadOnlyList<float> array2)
        {
            return Math.Sqrt(array1.Select((t, element) => Math.Pow(t - array2[element], 2)).Sum());
        }
        
        public static float Deviation(float[] valueList)
        {
            var mean = valueList.Sum() / valueList.Length;
            var squaresQuery = from float value in valueList select (value - mean) * (value - mean);
            var sumOfSquares = squaresQuery.Sum();
            return (float) Math.Sqrt(sumOfSquares / valueList.Length);
        }
        
        public static int GetHashFromArray<T>(IReadOnlyCollection<T> array)
        {
            unchecked
            {
                return array.Aggregate(array.Count, (current, element) => current * 31 + element.GetHashCode());
            }
        }
    }
}
