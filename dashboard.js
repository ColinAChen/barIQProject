import React from 'react';
import "../CSS/dashboard.css";
import img1 from './image.png';

const Dashboard = () => {
return (
	<div className='background'>
		<div id='header'>
			<p>Overview</p>
		</div>
		<div className='inventory'>
			<div className='box'>
				<p>Total Inventory</p>
				<p>647</p>
			</div>
			<div className='box'>
				<p>Brand 1</p>
				<p>420</p>
			</div>
			<div className='box'>
				<p>Brand 2</p>
				<p>158</p>
			</div>
			<div className='box'>
				<p>Brand 3</p>
				<p>69</p>
			</div>
		</div>
		<div className='chartbox'>
			<div className='chart'>
				<div className='title'>
					<p>Daily and Monthly Trends</p>
					<p>as of 30 January 2021, 4:20pm</p>
				</div>
				<img src={img1}></img>
			</div>
			<div id='chartinfo'>
				<div className='info'>
					<p>Pallets Received This Month</p>
					<p>417</p>
					
				</div>
				<div className='info'>
					<p>Pallets Received Today</p>
					<p>67</p>
				</div>
				<div className='info'>
					<p>Incoming Pallets</p>
					<p>20</p>
				</div>
				<div className='info'>
					<p>Outgoing Pallets</p>
					<p>48</p>
				</div>
				<div className='info'>
					<p>Accounted Cases Percentage</p>
					<p>98%</p>
				</div>
			</div>
			
		</div>
		<table id='analytics'>
			<tr>
				<th>Brand</th>
				<th>Pallets Received Today</th>
				<th>Pallets Inbound</th>
				<th>Pallets Outbound</th>
			</tr>
			<tr>
				<td>Brand</td>
				<td>Pallets Received Today</td>
				<td>Pallets Inbound</td>
				<td>Pallets Outbound</td>
			</tr>
			<tr>
				<td>Brand</td>
				<td>Pallets Received Today</td>
				<td>Pallets Inbound</td>
				<td>Pallets Outbound</td>
			</tr>
			<tr>
				<td>Brand</td>
				<td>Pallets Received Today</td>
				<td>Pallets Inbound</td>
				<td>Pallets Outbound</td>
			</tr>
		</table>
	</div>
);
};

export default Dashboard;