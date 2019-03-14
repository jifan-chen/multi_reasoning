import React from 'react';
import Collapsible from 'react-collapsible'
import './style.css'


class Tok extends React.Component {
  render() {
    return (
        <span>
        <span 
            className={this.props.class} 
            data-balloon={this.props.tip} 
            data-balloon-pos="up">
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
        <span className={this.props.class} data-balloon={this.props.tip} data-balloon-pos="auto">
            {this.props.tokens.map((tok, i) =>
                {
                    return <Tok token={tok} key={i}/>
                })
            }
        </span>
    );
  }
}

class Doc extends React.Component {
  render() {
    return (
        <div className="doc">
            {this.props.sent_spans.map((sp, i) =>
                {
                    var s = sp[0];
                    var e = sp[1] + 1;
                    var sent_tokens = this.props.tokens.slice(s, e);
                    var className = null;
                    if(this.props.pred_sent_labels[i] === 1 && this.props.sent_labels[i] === 1) {
                        className = "corr_support";
                    } else if(this.props.pred_sent_labels[i] === 1) {
                        className = "pred_support";
                    } else if(this.props.sent_labels[i] === 1) {
                        className = "support";
                    }
                    var sent_ele = <Sent class={className}
                                    tokens={sent_tokens}
                                    tip={this.props.pred_sent_probs[i]}
                                    key={i}
                                />;
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

export default class EvdVisualization extends React.Component {
  render() {
    return (
      <Collapsible trigger="Evidence Prediction">
          <Doc sent_spans={this.props.sent_spans}
               sent_labels={this.props.sent_labels}
               pred_sent_labels={this.props.pred_sent_labels}
               pred_sent_probs={this.props.pred_sent_probs}
               tokens={this.props.doc}
          />
      </Collapsible>
    );
  }
}

