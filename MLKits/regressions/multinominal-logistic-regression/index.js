require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const LogisticRegression = require('./logistic-regression');
const plot = require('node-remote-plot');
const _ = require('lodash');
const mnist = require('mnist-data');

const mnistData = mnist.training(0, 10);

const features = mnistData.images.values.map(image => _.flatMap(image));
const encodedLabels = mnistData.labels.values.map(label => {
  const row = new Array(10).fill(0);
  row[label] = 1;
  return row;
});

console.log(encodedLabels);
