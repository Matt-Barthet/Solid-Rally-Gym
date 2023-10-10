using System.Collections.Generic;
using System.IO;
using MoreMountains.HighroadEngine;
using UnityEditor;
using UnityEngine;
using UnityEngine.Scripting;

namespace Go_Blend.Scripts
{
    [System.Serializable]
    public class PromptData
    {
        public string source;
        public string target;
        public string prompt;
    }
    
    public class F2Experiment : MonoBehaviour
    {
        public Camera participantCamera;
        private SolidController carController;
        public int captureFrequency = 20; // capture every 20 frames
        public string sourceFolder = "source";
        public string targetFolder = "target";
        private static int frameId = 50000 - 4;
        private static bool shifted;
        private RenderTexture renderTexture;

        public int carID;
        
        private void Start()
        {
            // Create the source and target folders if they don't exist
            Directory.CreateDirectory(sourceFolder);
            Directory.CreateDirectory(targetFolder);

            // Create a RenderTexture object to capture the camera's contents
            renderTexture = new RenderTexture(960, 600, 24);
            participantCamera.targetTexture = renderTexture;

            carController = GetComponent<SolidController>();
            promptData = new PromptData();
            texture = new Texture2D(renderTexture.width, renderTexture.height, TextureFormat.RGB24, false);
            
            // Capture the first frame and save it in the source folder
            CaptureFrame(sourceFolder, frameId + carID);
            shifted = true;
            File.WriteAllText("prompt.json", "");
            Time.captureFramerate = 40;
        }

        private PromptData promptData;
        
        private void FixedUpdate()
        {
            if (Time.frameCount % captureFrequency != 0)
            {
                if (shifted) return;
                frameId += 4;
                shifted = true;
                return;
            }
            
            shifted = false;
            CaptureFrame(targetFolder, frameId + carID);

            // Generate the prompt text based on the input
            var prompt = GeneratePrompt();

            // Create a JSON object with the source file path, target file path, and prompt text
            promptData.source = Path.Combine(sourceFolder, frameId + carID + ".jpg");
            promptData.target = Path.Combine(targetFolder, frameId + carID + ".jpg");
            promptData.prompt = prompt;

            print(name + "," + JsonUtility.ToJson(promptData, false));
            
            var jsonData = JsonUtility.ToJson(promptData, false);
            File.AppendAllText("prompt.json", $"{jsonData}\n");
            
            CaptureFrame(sourceFolder, frameId + carID + 4);
        }

        string GeneratePrompt()
        {
            // Generate a prompt based on the input
            string speed;
            string direction;

            speed = carController.Speed switch
            {
                < 10 => "stationary",
                <= 30 => "moving forward slowly",
                > 30 => "moving forward quickly",
            };

            if (carController.CurrentSteeringAmount == 1)
            {
                direction = "turning right";
            } else if (carController.CurrentSteeringAmount == -1)
            {
                direction = "turning left";
            }
            else
            {
                direction = "keeping straight";
            }

            return $"A driving game where the car is {speed} and the driver is {direction}.";
        }

        private Texture2D texture;
        
        void CaptureFrame(string folderPath, int id)
        {
            // Capture the frame and save it in the specified folder with the specified ID
            RenderTexture.active = renderTexture;
            texture.ReadPixels(new Rect(0, 0, renderTexture.width, renderTexture.height), 0, 0);
            texture.Apply();
            var bytes = texture.EncodeToJPG();
            var fileName = Path.Combine(folderPath, id + ".jpg");
            File.WriteAllBytes(fileName, bytes);
            // Debug.Log("Saved frame: " + fileName);
        }
    }
}
