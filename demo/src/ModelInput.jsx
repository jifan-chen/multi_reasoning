import React from 'react';
import Button from './components/Button'
import ModelIntro from './components/ModelIntro'


// TODO: These are some quickly-accessible examples to try out with your model.  They will get
// added to the select box on the demo page, and will auto-populate your input fields when they
// are selected.  The names here need to match what's read in `handleListChange` below.

const examples = [
  {
    dataset: "train",
    instance_idx: 0,
    th: 0.2
  },
  {
    dataset: "dev",
    instance_idx: 1000,
    th: 0.1
  },
  {
    dataset: "train",
    instance_idx: 10000,
    th: 0.3
  }
];

// TODO: This determines what text shows up in the select box for each example.  The input to
// this function will be one of the items from the `examples` list above.
function summarizeExample(example) {
  return example.instance_idx;
}

// TODO: You can give a model name and description that show up in your demo.
const title = "Hotpot Predictor";
const description = (
  <span>
  </span>
);

class ModelInput extends React.Component {
  constructor(props) {
    super(props);
    this.handleListChange = this.handleListChange.bind(this);
    this.onClick = this.onClick.bind(this);
  }

  handleListChange(e) {
    if (e.target.value !== "") {
      // TODO: This gets called when the select box gets changed.  You want to set the values of
      // your input boxes with the content in your examples.
      this.instance_idx.value = examples[e.target.value].instance_idx
      this.dataset.value = examples[e.target.value].dataset
      this.th.value = examples[e.target.value].th
    }
  }

  onClick() {
    const { runModel } = this.props;

    // TODO: You need to map the values in your input boxes to json values that get sent to your
    // predictor.  The keys in this dictionary need to match what your predictor is expecting to receive.
    runModel({instance_idx: this.instance_idx.value,
        dataset: this.dataset.value,
        th: this.th.value});
  }

  render() {

    const { outputState } = this.props;

    return (
      <div className="model__content">
        <ModelIntro title={title} description={description} />
        <div className="form__instructions"><span>Enter text or</span>
          <select disabled={outputState === "working"} onChange={this.handleListChange}>
              <option value="">Choose an example...</option>
              {examples.map((example, index) => {
                return (
                    <option value={index} key={index}>{summarizeExample(example) + "..."}</option>
                );
              })}
          </select>
        </div>

       {/*
         * TODO: This is where you add your input fields.  You shouldn't have to change any of the
         * code in render() above here.  We're giving a couple of example inputs here, one for a
         * larger piece of text, like a paragraph (the `textarea`) and one for a shorter piece of
         * text, like a question (the `input`).  You'll probably want to change the variable names
         * here to match the input variable names in your model.
         */}

        <div className="form__field">
          <label>Dataset(train/dev)</label>
          <textarea ref={(x) => this.dataset = x} type="text" autoFocus="true"></textarea>
        </div>
        <div className="form__field">
          <label>Attention Display Threshold</label>
          <textarea ref={(x) => this.th = x} type="text" autoFocus="true"></textarea>
        </div>
        <div className="form__field">
          <label>Instance Idx from Hotpot Dataset</label>
          <textarea ref={(x) => this.instance_idx = x} type="text" autoFocus="true"></textarea>
        </div>

       {/* You also shouldn't have to change anything below here. */}

        <div className="form__field form__field--btn">
          <Button enabled={outputState !== "working"} onClick={this.onClick} />
        </div>
      </div>
    );
  }
}

export default ModelInput;
