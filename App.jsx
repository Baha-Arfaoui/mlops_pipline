import { useState } from "react";
import axios from "axios";
import "./App.css";

function App() {
  const [sender, setSender] = useState("");
  const [subject, setSubject] = useState("");
  const [body, setBody] = useState("");
  const [response, setResponse] = useState("");

  const loadSampleBusinessInquiry = () => {
    setSender("john.doe@company.com");
    setSubject("Business Collaboration Inquiry");
    setBody("Hello, we would like to discuss a potential business partnership.");
  };

  const loadSampleSpamEmail = () => {
    setSender("winner@lottery.com");
    setSubject("You won a prize!");
    setBody("Congratulations! Click here to claim your reward now.");
  };

  const handleSubmit = async () => {
    try {
      const res = await axios.post("https://your-api-url.com/api/email", {
        sender,
        subject,
        body,
      });
      setResponse(JSON.stringify(res.data, null, 2));
    } catch (err) {
      setResponse("Error: " + err.message);
    }
  };

  return (
    <div className="app">
      <div className="left-panel">
        <h2>BNP PARIBAS ASSET MANAGEMENT</h2>
        <h3>Alfred Email Bulter</h3>
        <p>
          Alfred is your personal email butler, powered by AI to help process,
          classify and respond to emails efficiently
        </p>
        <div className="demo-buttons">
          <button onClick={loadSampleBusinessInquiry}>
            Load Sample Business Inquiry
          </button>
          <button onClick={loadSampleSpamEmail}>
            Load Sample Spam Email
          </button>
        </div>
      </div>
      <div className="right-panel">
        <h2>Compose or Load an Email</h2>
        <input
          type="text"
          placeholder="Sender"
          value={sender}
          onChange={(e) => setSender(e.target.value)}
        />
        <input
          type="text"
          placeholder="Subject"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
        />
        <textarea
          placeholder="Email Body"
          value={body}
          onChange={(e) => setBody(e.target.value)}
        />
        <button onClick={handleSubmit}>Submit Email</button>
        {response && (
          <div className="response-box">
            <h4>Response:</h4>
            <pre>{response}</pre>
          </div>
        )}
      </div>
    </div>
  );
}

export default App;


.app {
  display: flex;
  font-family: Arial, sans-serif;
  height: 100vh;
  padding: 20px;
}

.left-panel {
  width: 30%;
  padding: 20px;
  border-right: 1px solid #ccc;
}

.right-panel {
  width: 70%;
  padding: 20px;
}

input,
textarea {
  display: block;
  width: 100%;
  margin-bottom: 10px;
  padding: 10px;
  font-size: 1rem;
}

textarea {
  height: 150px;
}

button {
  padding: 10px 15px;
  margin-right: 10px;
  cursor: pointer;
}
.response-box {
  margin-top: 20px;
  background-color: #f4f4f4;
  padding: 15px;
  border: 1px solid #ccc;
}

index 
/* src/index.css */

/* Reset some default browser styles */
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: Arial, sans-serif;
  background-color: #fdfdfd;
  color: #333;
  line-height: 1.5;
}

/* Scrollbar styling (optional) */
::-webkit-scrollbar {
  width: 8px;
}

::-webkit-scrollbar-thumb {
  background: #ccc;
  border-radius: 4px;
}

a {
  text-decoration: none;
  color: inherit;
}

button {
  font-family: inherit;
}


