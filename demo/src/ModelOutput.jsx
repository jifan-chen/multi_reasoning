import React from 'react';
//import HeatMap from './components/heatmap/HeatMap'
import Collapsible from 'react-collapsible'
import './balloon.css'
import './style.css'


class Tok extends React.Component {
  render() {
    return (
        <span>
        <span className={this.props.class} data-balloon={this.props.tip} data-balloon-pos="up">
            {this.props.token}
        </span>
        <span> </span>
        </span>
    );
  }
}

class Sent extends React.Component {
  render() {
    return (
        <span className={this.props.class}>
            {this.props.tokens.map((tok, i) =>
                {
                    var className = null;
                    var tip = null;
                    if(this.props.scores[i] > 0) {
                        className = "attn";
                        tip = this.props.scores[i];
                    } else if(i === this.props.pos) {
                        className = "target"
                        tip = this.props.type;
                    }
                    return <Tok class={className} token={tok} tip={tip} key={i}/>})
            }
        </span>
    );
  }
}

class Doc extends React.Component {
  render() {
    var cur_offset = 0;
    return (
        <div className="doc">
            {this.props.sent_spans.map((sp, i) =>
                {
                    var s = sp[0];
                    var e = sp[1] + 1;
                    var sent_tokens = this.props.tokens.slice(s, e);
                    var sent_scores = this.props.scores.slice(s, e);
                    var pos = this.props.pos - cur_offset;
                    console.log(cur_offset)
                    var className = null;
                    if(this.props.sent_labels[i] === 1) {
                        className = "support";
                    }
                    var sent_ele = <Sent class={className}
                                    tokens={sent_tokens}
                                    scores={sent_scores}
                                    pos={pos}
                                    type={this.props.type}
                                    key={i}
                                />;
                    cur_offset = cur_offset + e - s;
                    return sent_ele})
            }
            <div>
                <hr
                    style={{
                    border: '1px solid rgb(200, 200, 200)',
                    backgroundColor: 'rgb(200, 200, 200)'
                    }}
                />
            </div>
        </div>
    );
  }
}

class HeadAtt extends React.Component {
  render() {
    var name = "Head"+this.props.h_idx.toString();
    return (
      <Collapsible trigger={name}>
          {this.props.attns.map((tgt_dict, i) =>
              {
                  return (
                      <Doc sent_spans={this.props.sent_spans}
                           sent_labels={this.props.sent_labels}
                           tokens={this.props.doc}
                           scores={tgt_dict.scores}
                           pos={tgt_dict.pos}
                           type={tgt_dict.type}
                           key={i} />
                  )
              })
          }
      </Collapsible>
    );
  }
}

class ModelOutput extends React.Component {
  render() {

    const { outputs } = this.props;

    // TODO: `outputs` will be the json dictionary returned by your predictor.  You can pull out
    // whatever you want here and visualize it.  We're giving some examples of different return
    // types you might have.  Change names for data types you want, and delete anything you don't
    // need.
    // var string_result_field = outputs['string_result_field'];
    // This is a 1D attention array, which we need to make into a 2D matrix to use with our heat
    // map component.
    // var attention_data = outputs['attention_data'].map(x => [x]);
    // This is a 2D attention matrix.
    // var matrix_attention_data = outputs['matrix_attention_data'];
    // Labels for our 2D attention matrix, and the rows in our 1D attention array.
    // var column_labels = outputs['column_labels'];
    // var row_labels = outputs['row_labels'];

    // This is how much horizontal space you'll get for the row labels.  Not great to have to
    // specify it like this, or with this name, but that's what we have right now.
    // var xLabelWidth = "70px";
    var ins = outputs
    console.log(ins);

    return (
      <div className="model__content">

       {/*
         * TODO: This is where you display your output.  You can show whatever you want, however
         * you want.  We've got a few examples, of text-based output, and of visualizing model
         * internals using heat maps.
         */}

        <div className="form__field">
          <div className="model__content__summary">
              <div className="title">
                  <span className="title">Question:</span> {ins['question']} <br/>
                  <span className="title">Answer:</span> {ins['answer']} <br/>
                  <span className="title">Predict:</span> {ins['predict']} <br/>
                  <span className="title">F1:</span> {ins['f1']}
              </div>
          </div>
        </div>
        
        <div className="form__field">
          <Collapsible trigger="Examples of learned attentions">
              {ins['attns'].map((head_attns, i) =>
                  {
                      return (
                          <HeadAtt 
                               sent_spans={ins['sent_spans']}
                               sent_labels={ins['sent_labels']}
                               doc={ins['doc']}
                               attns={head_attns}
                               h_idx={i} 
                               key={i} />
                      )
                  })
              }
          </Collapsible>
        </div>

      </div>
    );
  }
}

export default ModelOutput;
