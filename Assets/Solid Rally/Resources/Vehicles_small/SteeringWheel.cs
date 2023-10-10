using MoreMountains.HighroadEngine;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SteeringWheel : MonoBehaviour
{
    public SolidController car;

    void Update() {
        transform.localEulerAngles = new Vector3(0, 0, -60 * car.CurrentSteeringAmount);
    }
}
