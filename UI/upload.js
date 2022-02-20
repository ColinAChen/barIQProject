import React, {useState} from 'react';
//import axios from 'axios';
import "../CSS/upload.css";

const Upload = () => {

  const [file, setFile] = useState()

  function handleChange(event) {
    setFile(event.target.files[0])
  }
  
  function handleSubmit(event) {
    event.preventDefault()
    const url = 'http://127.0.0.1:5000/';
    const formData = new FormData();
    formData.append('file', file);
    formData.append('fileName', file.name);
    const config = {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    };
    fetch(url, {
		  method: 'POST', headers: config.headers, body: formData
	  })
	  .then((response) => {
      console.log(response.json())
	  })
	  .catch(error => {
		  console.log(error.response.data)
    });

  }

  return (
    <div className="container">
        <form onSubmit={handleSubmit}>
          <h1>React File Upload</h1>
          <input type="file" onChange={handleChange}/>
          <button type="submit">Upload</button>
        </form>
    </div>
  );
}

export default Upload;