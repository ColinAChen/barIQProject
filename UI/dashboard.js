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
			</div>
			<div className='box'>
				<p>Brand 1</p>
			</div>
			<div className='box'>
				<p>Brand 2</p>
			</div>
			<div className='box'>
				<p>Brand 3</p>
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
					
				</div>
				<div className='info'>
					<p>Pallets Received Today</p>
				
				</div>
				<div className='info'>
					<p>Incoming Pallets</p>
					
				</div>
				<div className='info'>
					<p>Outgoing Pallets</p>
					
				</div>
				<div className='info'>
					<p>Accounted Cases Percentage</p>
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
