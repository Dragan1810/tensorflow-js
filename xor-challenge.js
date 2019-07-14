const trainingData = [];

const model = tf.sequential({
  layers: [
    tf.layers.dense({ inputShape: [784], units: 32, activation: 'relu' }),
    tf.layers.dense({ units: 10, activation: 'softmax' })
  ]
});
