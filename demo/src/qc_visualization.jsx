import React from 'react';
import Collapsible from 'react-collapsible'
import HeatMap from './allenlp_src/HeatMap.js'
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
    if(this.props.bold) {
        token_className = "title";
    }
    return (
        <span className={this.props.class}
              onMouseOver={mouse_over_func}
              onMouseOut={mouse_out_func}
              onClick={click_func}
              style={this.props.style}>
            {this.props.tokens.map((tok, i) =>
                {
                    return <Tok 
                                class={token_className} 
                                token={tok} 
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
                    var style = null;
                    var bold = false;
                    if(this.props.sent_labels[i] === 1) {
                        className = "support";
                    }
                    if(this.props.click_sent === i) {
                        style={
                            backgroundColor: '#bfeeb7'
                        }
                    }
                    if(this.props.act_sent === i) {
                        bold = true;
                    }
                    var sent_ele = <Sent class={className}
                                    bold={bold}
                                    tokens={sent_tokens}
                                    onMouseOver={() => mouse_over_func(i)}
                                    onMouseOut={() => mouse_out_func(i)}
                                    onClick={() => click_func(i)}
                                    style={style}
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

export default class QCVisualization extends React.Component {
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
    var act_sent = this.state.activeSentId;
    var click_sent = this.state.clickSentId;
    if(click_sent === 0 || click_sent) {
        var click_sent_sp = this.props.sent_spans[click_sent];
        var slice_rowLabels = this.props.rowLabels.slice(click_sent_sp[0], click_sent_sp[1]+1)
        var slice_data = this.props.data.slice(click_sent_sp[0], click_sent_sp[1]+1)
        return (
          <Collapsible trigger={this.props.name+" Attention"}>
              <Doc sent_spans={this.props.sent_spans}
                   sent_labels={this.props.sent_labels}
                   tokens={this.props.doc}
                   act_sent={act_sent}
                   click_sent={click_sent}
                   onMouseOver={this.handleMouseOver}
                   onMouseOut={this.handleMouseOut}
                   onClick={this.handleOnClick}
              />
              <HeatMap
                  colLabels={this.props.colLabels} 
                  rowLabels={slice_rowLabels}
                  data={slice_data} 
                  includeSlider={this.props.includeSlider} 
                  showAllCols={this.props.showAllCols} 
              />
          </Collapsible>
        );
    } else {
        return (
          <Collapsible trigger={this.props.name+" Attention"}>
              <Doc sent_spans={this.props.sent_spans}
                   sent_labels={this.props.sent_labels}
                   tokens={this.props.doc}
                   act_sent={act_sent}
                   click_sent={click_sent}
                   onMouseOver={this.handleMouseOver}
                   onMouseOut={this.handleMouseOut}
                   onClick={this.handleOnClick}
              />
          </Collapsible>
        );
    }
  }
}

