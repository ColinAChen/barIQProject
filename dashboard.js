import React from 'react';
import {useState} from 'react';
import "../CSS/dashboard.css";

const Dashboard = () => {

	const [total, setTotal] = useState();
	const [brand1, setBrand1] = useState();
	const [brand2, setBrand2] = useState();
	const [brand3, setBrand3] = useState();

	// function handleChange() {
	// 	const response = fetch('http://127.0.0.1:5000/');
	// 	const dat = response.json;
	// 	setTotal(dat.data.total);
	// 	setBrand1(dat.data.brand1);
	// 	setBrand2(dat.data.brand2);
	// 	setBrand3(dat.data.brand3);
	// 	console.log(total)
	// }

	function handleChange() {
		
		var axios = require('axios');

		var config = {
			method: 'get',
			url: 'http://127.0.0.1:5000/test'
		};
		
		axios(config)
		.then(function (response) {
			console.log(JSON.stringify(response.data));
			setTotal(JSON.stringify(response.data.total));
		})
		.catch(function (error) {
			console.log(error);
		});
	}
		
	// 	// fetch('http://127.0.0.1:5000/', {
	// 	// 	headers : { 
	// 	// 		'Content-Type': 'application/json',
	// 	// 		'Accept': 'application/json'
	// 	// 	   }
	// 	// })
	// 	// .then((response) => (response.json()))
	// 	// .then((response) => {
	// 	// 	console.log(response.json());
	// 	// })
	// 	// .catch((error) => {
	// 	// 	console.error(error);
	// 	// });
	// }

	return (
		<div className='background'>
			<div id='header'>
				<div id='title'>
					<p>Overview</p>
				</div>
				<div id='upload'>
					<button
						type="button"
						onClick={(e) => {
						e.preventDefault();
						window.location.href='http://127.0.0.1:5000/';
						}}
					> Upload Image</button>
					<button
						type="button"
						onClick={handleChange}
					> Update Page</button>
					<p></p>
				</div>
			</div>
			<div className='inventory'>
				<div className='box'>
					<div className='text'>
						<p>Total Inventory</p>
						<p>{total}</p>
					</div>
				</div>
				<div className='box'>
					<div className='text'>
						<p>Brand 1</p>
						<p>{brand1}</p>
					</div>
				</div>
				<div className='box'>
					<div className='text'>
						<p>Brand 2</p>
						<p>{brand2}</p>
					</div>
				</div>
				<div className='box'>
					<div className='text'>
						<p>Brand 3</p>
						<p>{brand3}</p>
					</div>
				</div>
			</div>
			<p></p>
			<p></p>
			<table id='analytics'>
				<tr>
					<th>Brand</th>
					<th>Total Pallets</th>
					<th>Pallets Inbound</th>
					<th>Pallets Outbound</th>
				</tr>
				<tr>
					<td>Brand</td>
					<td>Total Pallets</td>
					<td>Pallets Inbound</td>
					<td>Pallets Outbound</td>
				</tr>
				<tr>
					<td>Brand</td>
					<td>Total Pallets</td>
					<td>Pallets Inbound</td>
					<td>Pallets Outbound</td>
				</tr>
				<tr>
					<td>Brand</td>
					<td>Total Pallets</td>
					<td>Pallets Inbound</td>
					<td>Pallets Outbound</td>
				</tr>
				<tr>
					<td>Brand</td>
					<td>Total Pallets</td>
					<td>Pallets Inbound</td>
					<td>Pallets Outbound</td>
				</tr>
				<tr>
					<td>Brand</td>
					<td>Total Pallets</td>
					<td>Pallets Inbound</td>
					<td>Pallets Outbound</td>
				</tr>
			</table>
		</div>
	);
};

export default Dashboard;