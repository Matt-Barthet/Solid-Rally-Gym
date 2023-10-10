using System;
using Go_Blend.Scripts.Car_Controllers;
using Unity.Mathematics;
using Unity.MLAgents.SideChannels;
using UnityEngine;

namespace Go_Blend.Scripts.Utilities
{
    public class MySideChannel : SideChannel
    {
        private readonly GoBlendGymController agent;
        private GameObject currentTarget;
        private readonly Rigidbody rigidbody;

        public MySideChannel(GoBlendGymController pAgent, Rigidbody rigidbody)
        {
            ChannelId = new Guid("621f0a70-4f87-11ea-a6bf-784f4387d1f7");
            agent = pAgent;
            this.rigidbody = rigidbody;
        }

        protected override void OnMessageReceived(IncomingMessage incomingMessage)
        {
            var test = incomingMessage.ReadString();
            if (test.Contains("[Cell Name]:"))
            {
                var name = test.Split(":")[1];
                StateSaveLoad.LoadEnvironment(name, true);
            }
            else if (test.Contains("[Save Cells]:"))
            {
                Debug.LogError(bool.Parse(test.Split(":")[1]));
                agent.saveCells = bool.Parse(test.Split(":")[1]);
            }
            else if (test.Contains("[Target]:"))
            {
                var coordinates = test.Split(":")[1].Split(",");
                GameObject.Destroy(currentTarget);
                currentTarget = GameObject.Instantiate(
                    Resources.Load("Target Sphere"),
                    new Vector3(float.Parse(coordinates[0]), float.Parse(coordinates[1]), float.Parse(coordinates[2])),
                    quaternion.identity,
                    null
                ) as GameObject;
            }
            else if (test.Contains("[Generate Arousal]:"))
            {
                var value = bool.Parse(test.Split(":")[1]);
                agent.generateArousal = value;
            }
            else if (test.Contains("[Generate Saves]:"))
            {
                var value = bool.Parse(test.Split(":")[1]);
                agent.saveCells = value;
            } else if (test.Contains("[Set Position]:"))
            {
                var message = test.Split(":")[1];
                var positionString = message.Split("/")[0].Split(",");
                var rotationString = message.Split("/")[1].Split(",");
                var velocityString = message.Split("/")[2].Split(",");
                float[] position = { float.Parse(positionString[0]), float.Parse(positionString[1]),  float.Parse(positionString[2])};
                float[] rotation = { float.Parse(rotationString[0]), float.Parse(rotationString[1]),  float.Parse(rotationString[2])};
                float[] velocity = { float.Parse(velocityString[0]), float.Parse(velocityString[1]), float.Parse(velocityString[2])};
                rigidbody.MovePosition(new Vector3(position[0], position[1], position[2]));
                rigidbody.MoveRotation( Quaternion.Euler(rotation[0], rotation[1], rotation[2]));
                rigidbody.velocity = new Vector3(velocity[0], velocity[1], velocity[2]);
            } else if (test.Contains("[Save Dict]"))
            {
                Debug.Log("SAVING");
                HumanModel.SaveDict();
            } 
        }

        public void SendMessage(OutgoingMessage msg)
        {
            QueueMessageToSend(msg);
        }

    }
}
