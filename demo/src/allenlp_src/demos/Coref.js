import React from 'react';
//import { API_ROOT } from '../../api-config';
//import { withRouter } from 'react-router-dom';
//import Model from '../Model'
import HighlightContainer from '../highlight/HighlightContainer';
import { Highlight, getHighlightColor } from '../highlight/Highlight';
import Collapsible from 'react-collapsible'


// Helper function for transforming response data into a tree object
const transformToTree = (tokens, clusters) => {
  // Span tree data transform code courtesy of Michael S.
  function contains(span, index) {
    return index >= span[0] && index <= span[1];
  }

  let insideClusters = [
    {
      cluster: -1,
      contents: [],
      end: -1
    }
  ];

  tokens.forEach((token, i) => {
    // Find all the new clusters we are entering at the current index
    let newClusters = [];
    clusters.forEach((cluster, j) => {
      // Make sure we're not already in this cluster
      if (!insideClusters.map((c) => c.cluster).includes(j)) {
        cluster.forEach((span) => {
          if (contains(span, i)) {
              newClusters.push({ end: span[1], cluster: j });
          }
        });
      }
    });

    // Enter each new cluster, starting with the leftmost
    newClusters.sort(function(a, b) { return b.end - a.end }).forEach((newCluster) => {
      // Descend into the new cluster
      insideClusters.push(
        {
          cluster: newCluster.cluster,
          contents: [],
          end: newCluster.end
        }
      );
    });

    // Add the current token into the current cluster
    insideClusters[insideClusters.length-1].contents.push(token);

    // Exit each cluster we're at the end of
    while (insideClusters.length > 0 && insideClusters[insideClusters.length-1].end === i) {
      const topCluster = insideClusters.pop();
      insideClusters[insideClusters.length-1].contents.push(topCluster);
    }
  });

  return insideClusters[0].contents;
}

// Stateful
class Coref extends React.Component {
  constructor() {
    super();
    this.state = {
      selectedCluster: -1,
      activeIds: [],
      activeDepths: {ids:[],depths:[]},
      selectedId: null,
      isClicking: false
    };

    this.handleHighlightMouseDown = this.handleHighlightMouseDown.bind(this);
    this.handleHighlightMouseOver = this.handleHighlightMouseOver.bind(this);
    this.handleHighlightMouseOut = this.handleHighlightMouseOut.bind(this);
    this.handleHighlightMouseUp = this.handleHighlightMouseUp.bind(this);
  }

  handleHighlightMouseDown(id, depth) {
    let depthTable = this.state.activeDepths;
    depthTable.ids.push(id);
    depthTable.depths.push(depth);

    this.setState({
      selectedId: null,
      activeIds: [id],
      activeDepths: depthTable,
      isClicking: true
    });
  }

  handleHighlightMouseUp(id, prevState) {
    const depthTable = this.state.activeDepths;
    const deepestIndex = depthTable.depths.indexOf(Math.max(...depthTable.depths));

    this.setState(prevState => ({
      selectedId: depthTable.ids[deepestIndex],
      isClicking: false,
      activeDepths: {ids:[],depths:[]},
      activeIds: [...prevState.activeIds, id],
    }));
  }

  handleHighlightMouseOver(id, prevState) {
    this.setState(prevState => ({
      activeIds: [...prevState.activeIds, id],
    }));
  }

  handleHighlightMouseOut(id, prevState) {
    this.setState(prevState => ({
      activeIds: prevState.activeIds.filter(i => (i === this.state.selectedId)),
    }));
  }

  render() {
    const { activeIds, activeDepths, isClicking, selectedId } = this.state;

    const { responseData } = this.props
    const { document, clusters } = responseData

    const spanTree = transformToTree(document, clusters);

    // This is the function that calls itself when we recurse over the span tree.
    const spanWrapper = (data, depth) => {
      return data.map((token, idx) =>
        typeof(token) === "object" ? (
          <Highlight
            key={idx}
            activeDepths={activeDepths}
            activeIds={activeIds}
            color={getHighlightColor(token.cluster)}
            depth={depth}
            id={token.cluster}
            isClickable={true}
            isClicking={isClicking}
            label={token.cluster}
            labelPosition="left"
            onMouseDown={this.handleHighlightMouseDown}
            onMouseOver={this.handleHighlightMouseOver}
            onMouseOut={this.handleHighlightMouseOut}
            onMouseUp={this.handleHighlightMouseUp}
            selectedId={selectedId}>
            {/* Call Self */}
            {spanWrapper(token.contents, depth + 1)}
          </Highlight>
        ) : (
          <span key={idx}>{token} </span>
        )
      );
    }

    return (
      <div className="form__field">
        <Collapsible trigger={this.props.title}>
          <HighlightContainer isClicking={isClicking}>
            {spanWrapper(spanTree, 0)}
          </HighlightContainer>
        </Collapsible>
      </div>
    );
  }
}


export default Coref
