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
        {this.props.tip ?
            <ReactTooltip id={this.props.id}>
                {this.props.tip[0]}<br/>{this.props.tip[1]}
            </ReactTooltip> : null
        }
        </span>
    );
  }
}


class Chain extends React.Component {
  render() {
    var mouse_over_func = this.props.onMouseOver;
    var mouse_out_func = this.props.onMouseOut;
    var click_func = this.props.onClick;
    return (
        <div className="doc"
             onMouseOver={mouse_over_func}
             onMouseOut={mouse_out_func}
             onClick={click_func}>
            {this.props.chain.map((s_idx, i) =>
                {
                    var sp = this.props.sent_spans[s_idx];
                    var s = sp[0];
                    var e = sp[1] + 1;
                    var sent_tokens = this.props.tokens.slice(s, e);
                    var style = this.props.sent_style;
                    var sent_ele = <span key={i}>
                        <span className="title"> {i} {"(sent_idx: "+s_idx.toString()+"): "} </span>
                        <Sent
                            tokens={sent_tokens}
                            key={i}
                            style={style}
                        />
                        <br/>
                    </span>;
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


class Chains extends React.Component {
  render() {
    var mouse_over_func = this.props.onMouseOver;
    var mouse_out_func = this.props.onMouseOut;
    var click_func = this.props.onClick;
    var rb_sent_style = null;
    if(this.props.active_id === 0) {
        rb_sent_style={
            backgroundColor: '#e6f7e3'
        };
    }
    if(this.props.click_id === 0) {
        rb_sent_style={
            backgroundColor: '#bfeeb7'
        };
    }
    return (
        <div className="doc">
            <div className="title">
                <span className="title">Question:</span><span> {this.props.question}<br/> </span>
                <span className="title">Answer</span><span> {"\""+this.props.answer+"\""} is at {JSON.stringify(this.props.ans_sent_idxs)}<br/></span>
                <span className="title">Gold Evidence Set:</span><span> {JSON.stringify(this.props.gold_evd_set)}<br/> </span>
                <div>
                    <hr
                        style={{
                        border: '1px solid rgb(200, 200, 200)',
                        backgroundColor: 'rgb(200, 200, 200)'
                        }}
                    />
                </div>
            </div>
            <div><span className="title">Rule-Based Chain:</span></div>
            <Chain
                chain={this.props.rb_chain}
                sent_spans={this.props.sent_spans}
                tokens={this.props.tokens}
                onMouseOver={() => mouse_over_func(0)}
                onMouseOut={() => mouse_out_func(0)}
                onClick={() => click_func(0)}
                sent_style={rb_sent_style}
            />
            <div><span className="title">Predicted Chain:</span></div>
            {this.props.beam_pred_chains.map((chain, i) =>
                {
                    var sent_style = null;
                    if(this.props.active_id === i+1) {
                        //bold = true;
                        sent_style={
                            backgroundColor: '#e6f7e3'
                        };
                    }
                    if(this.props.click_id === i+1) {
                        sent_style={
                            backgroundColor: '#bfeeb7'
                        };
                    }
                    return <Chain
                                chain={chain}
                                sent_spans={this.props.sent_spans}
                                tokens={this.props.tokens}
                                key={i}
                                onMouseOver={() => mouse_over_func(i+1)}
                                onMouseOut={() => mouse_out_func(i+1)}
                                onClick={() => click_func(i+1)}
                                sent_style={sent_style}
                            />
                })
            }
        </div>
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
                    //var tip_text = [this.props.pred_sent_probs[i].toString()];
                    //if(this.props.att_score) {
                    //    tip_text.push("Heads: " + this.props.att_score[i].toString());
                    //}
                    var sent_ele = <Sent class={className}
                                        tokens={sent_tokens}
                                        //tip={tip_text}
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

export default class ChainVisualization extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
        activeSentId: null,
        clickSentId: null,
        activeChainId: null,
        clickChainId: null,
    };

    this.handleMouseOver = this.handleMouseOver.bind(this);
    this.handleMouseOut = this.handleMouseOut.bind(this);
    this.handleOnClick = this.handleOnClick.bind(this);

    this.handleChainMouseOver = this.handleChainMouseOver.bind(this);
    this.handleChainMouseOut = this.handleChainMouseOut.bind(this);
    this.handleChainOnClick = this.handleChainOnClick.bind(this);

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

  handleChainMouseOver(ChainId) {
    this.setState({
        activeChainId: ChainId,
    });
  }

  handleChainMouseOut() {
    this.setState({
        activeChainId: null,
    });
  }

  handleChainOnClick(ChainId) {
    this.setState({
        clickChainId: ChainId,
    });
  }

  render() {
    var gold_evd_set = [];
    var i;
    for(i = 0; i < this.props.sent_labels.length; i++) {
        if(this.props.sent_labels[i] === 1) {
            gold_evd_set.push(i)
        }
    }
    var beam_pred_chains = this.props.beam_pred_chains;


    var chain_click_id = this.state.clickChainId;
    var chain_active_id = this.state.activeChainId;

    var beam_pred_sent_orders = this.props.pred_sent_orders;
    var pred_sent_orders = null;
    if(chain_click_id === 0 || chain_click_id) {
        if(chain_click_id === 0) {
            //means mouse on the rb chain
            pred_sent_orders = new Array(this.props.sent_labels.length).fill(-1);
            for(i = 0; i < this.props.rb_chain.length; i++) {
                var s_idx = this.props.rb_chain[i]
                pred_sent_orders[s_idx] = i
            }
        } else {
            pred_sent_orders = beam_pred_sent_orders[chain_click_id-1]
        }
    }

    var pred_sent_labels = new Array(this.props.sent_labels.length).fill(0);
    if(pred_sent_orders) {
        for(i = 0; i < pred_sent_orders.length; i++) {
            pred_sent_labels[i] = (pred_sent_orders[i] >= 0) ? 1 : 0;
        }
    }
    console.log(pred_sent_orders);
    console.log(pred_sent_labels);


    var click_id = this.state.clickSentId;
    var active_id = this.state.activeSentId;

    var att_scores = this.props.att_scores;
    var att_score = (click_id === 0 || click_id) ? (att_scores ? att_scores[click_id] : null) : null;

    return (
      <Collapsible trigger="Chain Prediction">
          <Chains
               question={this.props.question}
               answer={this.props.answer}
               ans_sent_idxs={this.props.ans_sent_idxs}
               gold_evd_set={gold_evd_set}
               sent_spans={this.props.sent_spans}
               rb_chain={this.props.rb_chain}
               beam_pred_chains={beam_pred_chains}
               tokens={this.props.doc}
               active_id={chain_active_id}
               click_id={chain_click_id}
               onMouseOver={this.handleChainMouseOver}
               onMouseOut={this.handleChainMouseOut}
               onClick={this.handleChainOnClick}
          />
          <Doc sent_spans={this.props.sent_spans}
               sent_labels={this.props.sent_labels}
               pred_sent_labels={pred_sent_labels}
               pred_sent_orders={pred_sent_orders}
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

