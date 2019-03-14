import React from 'react';
//import HeatMap from './components/heatmap/HeatMap'
//import HeatMap from './allenlp_src/HeatMap.js'
//import Collapsible from 'react-collapsible'
import './style.css'
import HeadVisualization from './head_visualization.jsx'
import QCVisualization from './qc_visualization.jsx'
import EvdVisualization from './evd_visualization.jsx'
import Coref from './allenlp_src/demos/Coref.js'


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
    var coref_data = {"document": ins['doc'], "clusters": ins['coref_clusters']};
    var qc_scores = ins['qc_scores'];
    var qc_scores_sp = ins['qc_scores_sp'];
    var pred_sent_probs = ins['pred_sent_probs'];
    var pred_sent_labels = ins['pred_sent_labels'];
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
                  <span className="title">F1:</span> {ins['f1']} <br/>
                  {pred_sent_probs ? <span><span className="title">Evd Precision:</span> {ins['evd_measure']['prec']}<br/></span> : null}
                  {pred_sent_probs ? <span><span className="title">Evd Recall:</span> {ins['evd_measure']['recl']}<br/></span> : null}
                  {pred_sent_probs ? <span><span className="title">Evd F1:</span> {ins['evd_measure']['f1']}</span> : null}
              </div>
          </div>
        </div>
        
        <div className="form__field">
          <HeadVisualization 
               sent_spans={ins['sent_spans']}
               sent_labels={ins['sent_labels']}
               doc={ins['doc']}
               attns={ins['attns']}
          />
        </div>
        <div className="form__field">
          <Coref 
               responseData={coref_data}
          />
        </div>
        <div>
          <QCVisualization
              sent_spans={ins['sent_spans']}
              sent_labels={ins['sent_labels']}
              doc={ins['doc']}
              colLabels={ins['question_tokens']} 
              rowLabels={ins['doc']}
              data={qc_scores} 
              includeSlider={true} 
              showAllCols={true} 
              name="Passage-Question"
          />
        </div>
        { qc_scores_sp ?
            <div>
              <QCVisualization
                  sent_spans={ins['sent_spans']}
                  sent_labels={ins['sent_labels']}
                  doc={ins['doc']}
                  colLabels={ins['question_tokens']} 
                  rowLabels={ins['doc']}
                  data={qc_scores_sp} 
                  includeSlider={true} 
                  showAllCols={true} 
                  name="Evd Passage-Question"
              />
            </div> : null
        }
        { pred_sent_probs ?
            <div>
              <EvdVisualization
                  sent_spans={ins['sent_spans']}
                  sent_labels={ins['sent_labels']}
                  pred_sent_labels={pred_sent_labels}
                  pred_sent_probs={pred_sent_probs}
                  doc={ins['doc']}
              />
            </div> : null
        }
      </div>
    );
  }
}

export default ModelOutput;
