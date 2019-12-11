import React, { Component } from "react";
import Webcam from "react-webcam";
import "./App.css";
import { RPSDataset } from "./tfjs/data.js";
import { getAdvancedModel, getSimpleModel } from "./tfjs/models.js";
import { train } from "./tfjs/train.js";
import {
  showAccuracy,
  showConfusion,
  showExamples,
  doSinglePrediction
} from "./tfjs/evaluationHelpers.js";
import AdvancedModel from "./AdvancedModel.js";
import * as tfvis from "@tensorflow/tfjs-vis";
import * as tf from "@tensorflow/tfjs";

const DETECTION_PERIOD = 2000;

class App extends Component {
  state = {
    currentModel: null,
    webcamActive: false,
    camMessage: "",
    advancedDemo: false,
    loadDataMessage: "Load and Show Examples"
  };

  _renderAdvancedModel = () => {
    if (this.state.advancedDemo) {
      return (
        <div>
          <AdvancedModel key="advancedDemo" />
        </div>
      );
    }
  };

  componentDidMount() {
    /*
    Some code for debugging, sorrrrryyyyyy where is the best place for this?
    */
    window.tf = tf;
  }

  _renderWebcam = () => {
    if (this.state.webcamActive) {
      return (
        <div className="results">
          <div>64x64 Input</div>
          <canvas id="compVision" />
          <div>{this.state.camMessage}</div>
          <Webcam ref={this._refWeb} className="captureCam" />
        </div>
      );
    }
  };

  sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  detectWebcam = async () => {
    await this.sleep(100);
    const video = document.querySelectorAll(".captureCam");
    const feedbackCanvas = document.getElementById("compVision");
    // assure video is still shown
    if (video[0]) {
      const options = { feedbackCanvas };
      const predictions = await doSinglePrediction(
        this.model,
        video[0],
        options
      );
      const camMessage = predictions
        .map(p => ` ${p.className}: %${(p.probability * 100).toFixed(2)}`)
        .toString();
      this.setState({ camMessage });
      setTimeout(this.detectWebcam, DETECTION_PERIOD);
    }
  };

  _refWeb = webcam => {
    this.webcam = webcam;
  };

  render() {
    return (
      <div className="App">
        <div className="Main">
          <button
            className="myButton"
            onClick={async () => {
              this.setState({ loadDataMessage: "Loading 10MB Data" });
              const data = new RPSDataset();
              this.data = data;
              await data.load();
              await showExamples(data);
              this.setState({ loadDataMessage: "Data Loaded!" });
            }}
          >
            {this.state.loadDataMessage}
          </button>
          <div className="GroupUp">
            <button
              className={
                this.state.currentModel === "Simple"
                  ? "myButton activeModel"
                  : "myButton"
              }
              onClick={async () => {
                this.setState({ currentModel: "Simple" });
                const model = getSimpleModel();
                tfvis.show.modelSummary(
                  { name: "Simple Model Architecture" },
                  model
                );
                this.model = model;
              }}
            >
              Create Simple Model
            </button>
            <button
              className={
                this.state.currentModel === "Advanced"
                  ? "myButton activeModel"
                  : "myButton"
              }
              onClick={async () => {
                this.setState({ currentModel: "Advanced" });
                const model = getAdvancedModel();
                tfvis.show.modelSummary(
                  { name: "Advanced Model Architecture" },
                  model
                );
                this.model = model;
              }}
            >
              Create Advanced Model
            </button>
          </div>
          <p>
            Creating a model, is the structure and blueprint. It starts off able
            to, but terrible at predicting.
          </p>
          <button
            className="myButton"
            onClick={async () => {
              // stop errors
              if (!this.data) return;
              if (!this.model) return;
              await showAccuracy(this.model, this.data);
              await showConfusion(this.model, this.data, "Untrained Matrix");
            }}
          >
            Check Untrained Model Results
          </button>
          <button
            className="myButton"
            onClick={async () => {
              // stop errors
              if (!this.data) return;
              if (!this.model) return;
              const numEpochs = this.state.currentModel === "Simple" ? 12 : 20;
              await train(this.model, this.data, numEpochs);
            }}
          >
            Train Your {this.state.currentModel} Model
          </button>
          <p>
            It should be smarter at identifying RPS! We can now test it with 420
            RPS images it's never seen before.
          </p>
          <button
            className="myButton"
            onClick={async () => {
              // stop errors
              if (!this.data) return;
              if (!this.model) return;
              await showAccuracy(this.model, this.data, "Trained Accuracy");
              await showConfusion(
                this.model,
                this.data,
                "Trained Confusion Matrix"
              );
            }}
          >
            Check Model After Training
          </button>
          <button
            className="myButton"
            onClick={async () => {
              // stop errors
              if (!this.model) return;
              this.setState(
                prevState => ({
                  advancedDemo: false,
                  webcamActive: !prevState.webcamActive,
                  camMessage: ""
                }),
                this.detectWebcam
              );
            }}
          >
            {this.state.webcamActive ? "Turn Webcam Off" : "Launch Webcam"}
          </button>
          {this._renderWebcam()}
          <p>
            What does it look like to train a far more advanced model for hours
            that results in a 20+MB model? Here's an opportunity for you to try
            it yourself! This model isn't as diverse, but for demo purposes it's
            inspiring!
          </p>
          <button
            className="myButton"
            onClick={() => {
              this.setState(prevState => ({
                webcamActive: false,
                advancedDemo: !prevState.advancedDemo
              }));
            }}
          >
            {this.state.advancedDemo
              ? "Turn Off Advanced Demo"
              : "Show Advanced Demo"}
          </button>
          {this._renderAdvancedModel()}
        </div>
      </div>
    );
  }
}

export default App;
