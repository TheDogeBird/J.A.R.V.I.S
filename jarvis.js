class Neuron {
  constructor(numInputs) {
    this.numInputs = numInputs;
    this.weights = [];
    this.deltaWeights = [];

    for (let i = 0; i < numInputs; i++) {
      this.weights.push(Math.random() * 2 - 1);
      this.deltaWeights.push(0);
    }

    this.output = 0;
    this.delta = 0;
  }
}

class Layer {
  constructor(numNeurons, numInputsPerNeuron) {
    this.numNeurons = numNeurons;
    this.neurons = [];

    for (let i = 0; i < numNeurons; i++) {
      this.neurons.push(new Neuron(numInputsPerNeuron));
    }
  }
}

class JARVIS {
  constructor(numInputs, numOutputs, numHiddenLayers, numNeuronsPerHiddenLayer, activationFunction, derivativeActivationFunction) {
    this.numInputs = numInputs;
    this.numOutputs = numOutputs;
    this.numHiddenLayers = numHiddenLayers;
    this.numNeuronsPerHiddenLayer = numNeuronsPerHiddenLayer;
    this.activationFunction = activationFunction;
    this.derivativeActivationFunction = derivativeActivationFunction;
    this.layers = [];

    this.layers.push(new Layer(numNeuronsPerHiddenLayer, numInputs));

    for (let i = 0; i < numHiddenLayers - 1; i++) {
      this.layers.push(new Layer(numNeuronsPerHiddenLayer, numNeuronsPerHiddenLayer));
    }

    this.layers.push(new Layer(numOutputs, numNeuronsPerHiddenLayer));
  }

  forwardPropagate(input) {
    for (let i = 0; i < this.numInputs; i++) {
      this.layers[0].neurons[i].output = input[i];
    }

    for (let i = 1; i < this.numHiddenLayers + 1; i++) {
      const prevLayer = this.layers[i - 1];

      for (let j = 0; j < this.layers[i].numNeurons; j++) {
        let sum = 0;

        for (let k = 0; k < prevLayer.numNeurons; k++) {
          sum += prevLayer.neurons[k].output * this.layers[i].neurons[j].weights[k];
        }

        this.layers[i].neurons[j].output = this.activationFunction(sum);
      }
    }

    const output = [];

    for (let i = 0; i < this.numOutputs; i++) {
      output.push(this.layers[this.numHiddenLayers + 1].neurons[i].output);
    }

    return output;
  }

train(trainingData, options) {
  const { iterations, learningRate } = options;

  for (let i = 0; i < iterations; i++) {
    for (let j = 0; j < trainingData.length; j++) {
      const { input, output } = trainingData[j];

      this.forwardPropagate(input);
      
      for (let k = 0; k < this.layers[this.numHiddenLayers].numNeurons; k++) {
        const error = output[k] - this.layers[this.numHiddenLayers].neurons[k].output;
        this.layers[this.numHiddenLayers].neurons[k].delta = error * this.derivativeActivationFunction(this.layers[this.numHiddenLayers].neurons[k].output);

        for (let m = 0; m < this.layers[this.numHiddenLayers].neurons[k].numInputs; m++) {
          this.layers[this.numHiddenLayers].neurons[k].deltaWeights[m] = learningRate * this.layers[this.numHiddenLayers].neurons[k].delta * this.layers[this.numHiddenLayers - 1].neurons[m].output;
          this.layers[this.numHiddenLayers].neurons[k].weights[m] += this.layers[this.numHiddenLayers].neurons[k].deltaWeights[m];
        }
      }

      for (let k = this.numHiddenLayers - 1; k >= 0; k--) {
        const currentLayer = this.layers[k];
        const nextLayer = this.layers[k + 1];

        for (let m = 0; m < currentLayer.numNeurons; m++) {
          let error = 0;

          for (let n = 0; n < nextLayer.numNeurons; n++) {
            error += nextLayer.neurons[n].delta * nextLayer.neurons[n].weights[m];
          }

          currentLayer.neurons[m].delta = error * this.derivativeActivationFunction(currentLayer.neurons[m].output);

          for (let p = 0; p < currentLayer.neurons[m].numInputs; p++) {
            currentLayer.neurons[m].deltaWeights[p] = learningRate * currentLayer.neurons[m].delta * (k === 0 ? input[p] : this.layers[k - 1].neurons[p].output);
            currentLayer.neurons[m].weights[p] += currentLayer.neurons[m].deltaWeights[p];
          }
        }
      }
    }
  }
}
}


