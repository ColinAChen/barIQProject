import React, { Component } from 'react';
import LineChart from 'react-linechart';
import '../node_modules/react-linechart/dist/styles.css';
 
export default class Chart extends Component {
    render() {
      const  data=this.props.data
        return (
            <div>
                <div className="App">
                    <h1>My First LineChart</h1>
                    <LineChart 
                        width={600}
                        height={400}
                        data={data}
                    />
                </div>				
            </div>
        );
    }
}