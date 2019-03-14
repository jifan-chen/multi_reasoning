import React from 'react';
import Collapsible from 'react-collapsible'
import './balloon.css'
import './style.css'


class Tok extends React.Component {
  render() {
    return (
        <span>
        <span 
            className={this.props.class} 
            data-balloon={this.props.tip} 
            data-balloon-pos="up"
            onMouseOver={this.props.onMouseOver}
            onMouseOut={this.props.onMouseOut}
            onClick={this.props.onClick}>
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
    return (
        <span className={this.props.class}>
            {this.props.tokens.map((tok, i) =>
                {
                    var orig_i = i + this.props.offset;
                    var className = null;
                    var tip = null;
                    if(orig_i === this.props.pos) {
                        className = "title"
                        tip = this.props.mouse_type
                    }
                    if(this.props.scores && this.props.scores[i] > 0) {
                        className = "attn";
                        tip = this.props.scores[i];
                    } else if(orig_i === this.props.click_pos) {
                        className = "target"
                        tip = this.props.type;
                    }
                    return <Tok 
                                class={className} 
                                token={tok} 
                                tip={tip} 
                                onMouseOver={() => mouse_over_func(orig_i)}
                                onMouseOut={() => mouse_out_func()}
                                onClick={() => click_func(orig_i)}
                                key={i}
                            />
                })
            }
        </span>
    );
  }
}

class Doc extends React.Component {
  render() {
    var cur_offset = 0;
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
                    var sent_scores = this.props.scores ? this.props.scores.slice(s, e) : null;
                    var className = null;
                    if(this.props.sent_labels[i] === 1) {
                        className = "support";
                    }
                    var sent_ele = <Sent class={className}
                                    tokens={sent_tokens}
                                    scores={sent_scores}
                                    pos={this.props.pos}
                                    click_pos={this.props.click_pos}
                                    offset={cur_offset}
                                    type={this.props.type}
                                    mouse_type={this.props.mouse_type}
                                    onMouseOver={mouse_over_func}
                                    onMouseOut={mouse_out_func}
                                    onClick={click_func}
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
  constructor(props) {
    super(props);

    this.state = {
        activeTokId: null,
        clickTokId: null,
    };

    this.handleMouseOver = this.handleMouseOver.bind(this);
    this.handleMouseOut = this.handleMouseOut.bind(this);
    this.handleOnClick = this.handleOnClick.bind(this);
  }

  handleMouseOver(TokId) {
    this.setState({
        activeTokId: TokId,
    });
  }

  handleMouseOut() {
    this.setState({
        activeTokId: null,
    });
  }

  handleOnClick(TokId) {
    this.setState({
        clickTokId: TokId,
    });
  }

  findTgtDict(pos) {
    var attns = this.props.attns;
    var i;
    for(i in attns) {
        if(attns[i].pos === pos) {
            return attns[i];
        }
    }
    return null
  }

  render() {
    var name = "Head"+this.props.h_idx.toString();
    var click_pos = this.state.clickTokId;
    var tgt_dict = this.findTgtDict(click_pos)
    var scores = tgt_dict ? tgt_dict.scores : null;
    var type = tgt_dict ? tgt_dict.type : null;
    var pos = this.state.activeTokId;
    var mouse_tgt_dict = this.findTgtDict(pos)
    var mouse_type = mouse_tgt_dict ? mouse_tgt_dict.type : null;
    return (
      <Collapsible trigger={name}>
          <Doc sent_spans={this.props.sent_spans}
               sent_labels={this.props.sent_labels}
               tokens={this.props.doc}
               scores={scores}
               pos={pos}
               click_pos={click_pos}
               type={type}
               mouse_type={mouse_type}
               onMouseOver={this.handleMouseOver}
               onMouseOut={this.handleMouseOut}
               onClick={this.handleOnClick}
          />
      </Collapsible>
    );
  }
}

export default class HeadVisualization extends React.Component {
  render() {
    return (
      <Collapsible trigger="Examples of learned attentions">
          {this.props.attns.map((head_attns, i) =>
              {
                  return (
                      <HeadAtt 
                           sent_spans={this.props.sent_spans}
                           sent_labels={this.props.sent_labels}
                           doc={this.props.doc}
                           attns={head_attns}
                           h_idx={i} 
                           key={i} />
                  )
              })
          }
      </Collapsible>
    );
  }
}

