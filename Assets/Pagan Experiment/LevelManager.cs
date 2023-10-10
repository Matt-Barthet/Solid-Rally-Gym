using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.SceneManagement;
using UnityEngine.Events;
using MoreMountains.HighroadEngine;
using UnityEngine.Networking;
using System.Runtime.InteropServices;

public class LevelManager : MonoBehaviour
{
    public Text endCountdown;
    public int endCountdownTimer;
    public string nextLevel;
    public bool lastLevel;
    public MoreMountains.HighroadEngine.RaceManager raceManager;
    public bool recordLevel;

    void Start() {
        raceManager.OnGameEnds.AddListener(EndLevel);        
        if (recordLevel) {
            // raceManager.OnGameStarts.AddListener(StartRecord);
        }
    }

    private void EndLevel() {
        StartCoroutine(EndLevelProcess());
    }

    /// <summary>
    /// Starts the game shutdown coroutine.
    /// </summary>
    /// <returns>yield enumerator</returns>
    public IEnumerator EndLevelProcess() {
        Destroy(LocalLobbyManager.Instance.gameObject);
        if (!lastLevel) {
            for (int i = endCountdownTimer; i > -1; i--) {
                endCountdown.text = String.Format("Next level loads in {0} sec...", i);
                if (i <= 0) {
                    SceneManager.LoadScene(nextLevel);
                }
                yield return new WaitForSeconds(1f);
            }
        } else {
            for (int i = endCountdownTimer; i > -1; i--) {
                endCountdown.text = String.Format("Game ends in {0} sec...", i);
                if (i <= 0) {
                    if (recordLevel) {
                        endCountdown.text = "Please stand by... \n Your gameplay is being submitted...";
                    }
                }
                yield return new WaitForSeconds(1f);
            }
        }
    }
}
