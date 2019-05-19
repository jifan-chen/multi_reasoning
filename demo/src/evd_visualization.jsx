import React from 'react';
import ReactTooltip from 'react-tooltip'
import Collapsible from 'react-collapsible'
import { Highlight } from './allenlp_src/highlight/Highlight'
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
    var mouse_over_func = this.props.onMouseOver;
    var mouse_out_func = this.props.onMouseOut;
    var click_func = this.props.onClick;
    var token_className = null;
    //if(this.props.bold) {
    //    token_className = "title";
    //}
    return (
        <span>
        <span className={this.props.class} 
              //data-balloon={this.props.tip} 
              //data-balloon-pos="auto"
              data-tip
              data-for={this.props.id} 
              onMouseOver={mouse_over_func}
              onMouseOut={mouse_out_func}
              onClick={click_func}
              style={this.props.style}>
            {this.props.tokens.map((tok, i) =>
                {
                    return <Tok class={token_className} token={tok} key={i}/>
                })
            }
        </span>
        <ReactTooltip id={this.props.id}>
            {this.props.tip[0]}<br/>{this.props.tip[1]}
        </ReactTooltip>
        </span>
    );
  }
}

class Doc extends React.Component {
  render() {
    var mouse_over_func = this.props.onMouseOver;
    var mouse_out_func = this.props.onMouseOut;
    var click_func = this.props.onClick;
    return (
        <div className="doc">
            {this.props.sent_spans.map((sp, i) =>
                {
                    var s = sp[0];
                    var e = sp[1] + 1;
                    var sent_tokens = this.props.tokens.slice(s, e);
                    var className = null;
                    var style = this.props.att_score ? {
                            backgroundColor: `rgba(0, 0, 255, ${this.props.att_score[i][1]})`
                    } : null;
                    //var bold = false;
                    if(this.props.pred_sent_labels[i] === 1 && this.props.sent_labels[i] === 1) {
                        className = "corr_support";
                    } else if(this.props.pred_sent_labels[i] === 1) {
                        className = "pred_support";
                    } else if(this.props.sent_labels[i] === 1) {
                        className = "support";
                    }
                    if(this.props.active_id === i) {
                        //bold = true;
                        style={
                            backgroundColor: '#e6f7e3'
                        };
                    }
                    if(this.props.click_id === i) {
                        style={
                            backgroundColor: '#bfeeb7'
                        };
                    }
                    var tip_text = [this.props.pred_sent_probs[i].toString()];
                    if(this.props.att_score) {
                        tip_text.push("Heads: " + this.props.att_score[i].toString());
                    }
                    var sent_ele = <Sent class={className}
                                        tokens={sent_tokens}
                                        tip={tip_text}
                                        key={i}
                                        id={i.toString()}
                                        onMouseOver={() => mouse_over_func(i)}
                                        onMouseOut={() => mouse_out_func(i)}
                                        onClick={() => click_func(i)}
                                        style={style}
                                        //bold={bold}
                                    />;
                    if(this.props.pred_sent_orders && this.props.pred_sent_orders[i] >= 0) {
                        sent_ele = <Highlight
                                        key={i}
                                        id={this.props.pred_sent_orders[i]}
                                        label={this.props.pred_sent_orders[i]}
                                        labelPosition="left">
                                        {sent_ele}
                                   </Highlight>;
                    }
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
  constructor(props) {
    super(props);

    this.state = {
        activeSentId: null,
        clickSentId: null,
    };

    this.handleMouseOver = this.handleMouseOver.bind(this);
    this.handleMouseOut = this.handleMouseOut.bind(this);
    this.handleOnClick = this.handleOnClick.bind(this);

  }
  handleMouseOver(SentId) {
    this.setState({
        activeSentId: SentId,
    });
  }

  handleMouseOut() {
    this.setState({
        activeSentId: null,
    });
  }

  handleOnClick(SentId) {
    this.setState({
        clickSentId: SentId,
    });
  }

  render() {
    var click_id = this.state.clickSentId;
    var active_id = this.state.activeSentId;
    var att_scores = this.props.att_scores;
    var att_score = (click_id === 0 || click_id) ? (att_scores ? att_scores[click_id] : null) : null;
    return (
      <Collapsible trigger="Evidence Prediction">
          <Doc sent_spans={this.props.sent_spans}
               sent_labels={this.props.sent_labels}
               pred_sent_labels={this.props.pred_sent_labels}
               pred_sent_probs={this.props.pred_sent_probs}
               pred_sent_orders={this.props.pred_sent_orders}
               att_score={att_score}
               tokens={this.props.doc}
               active_id={active_id}
               click_id={click_id}
               onMouseOver={this.handleMouseOver}
               onMouseOut={this.handleMouseOut}
               onClick={this.handleOnClick}
          />
      </Collapsible>
    );
  }
}

