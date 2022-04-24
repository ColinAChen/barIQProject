import React from 'react';
import './App.css';
import Navbar from './components/Navbar';
import { BrowserRouter as Router, Routes, Route}
	from 'react-router-dom';
import Dashboard from './pages/dashboard';
import Upload from './pages/upload';

function App() {
return (
	<Dashboard></Dashboard>
);
}

export default App;

