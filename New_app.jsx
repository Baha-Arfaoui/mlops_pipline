import { useState } from 'react';

export default function App() {
  const [formData, setFormData] = useState({
    sender: '',
    subject: '',
    emailBody: ''
  });
  const [response, setResponse] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const loadSampleBusinessInquiry = () => {
    setFormData({
      sender: 'john.doe@company.com',
      subject: 'Partnership Inquiry - Strategic Collaboration',
      emailBody: 'Dear BNP Paribas Asset Management Team,\n\nI hope this email finds you well. I am writing to explore potential partnership opportunities between our organizations.\n\nOur company has been following BNP Paribas Asset Management\'s innovative approach to sustainable investing, and we believe there may be synergies worth exploring.\n\nWould it be possible to schedule a call to discuss this further?\n\nBest regards,\nJohn Doe\nBusiness Development Manager'
    });
    setResponse('');
  };

  const loadSampleSpamEmail = () => {
    setFormData({
      sender: 'noreply@suspicious-deals.com',
      subject: 'URGENT: Claim Your Prize NOW!!!',
      emailBody: 'CONGRATULATIONS!!! You have WON $1,000,000 in our international lottery!\n\nClick here immediately to claim your prize: www.fake-lottery-site.com\n\nThis offer expires in 24 hours! Act NOW!\n\nSend us your bank details and social security number to process your winnings.\n\nDon\'t miss this ONCE IN A LIFETIME opportunity!'
    });
    setResponse('');
  };

  const submitEmail = async () => {
    if (!formData.sender || !formData.subject || !formData.emailBody) {
      alert('Please fill in all fields');
      return;
    }

    setIsLoading(true);
    setResponse('');

    try {
      // Replace with your actual API endpoint
      const apiResponse = await fetch('/api/process-email', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          sender: formData.sender,
          subject: formData.subject,
          emailBody: formData.emailBody
        })
      });

      if (apiResponse.ok) {
        const result = await apiResponse.text();
        setResponse(result);
      } else {
        // Simulate a response for demo purposes
        const mockResponse = `Email Classification Results:

Sender: ${formData.sender}
Subject: ${formData.subject}

Analysis:
- Email Type: ${formData.subject.includes('URGENT') || formData.emailBody.includes('Click here') ? 'Potential Spam' : 'Business Communication'}
- Priority: ${formData.subject.includes('Partnership') || formData.subject.includes('Inquiry') ? 'High' : 'Low'}
- Action Required: ${formData.emailBody.includes('schedule a call') ? 'Response needed' : 'Review and file'}
- Confidence Score: ${Math.floor(Math.random() * 30) + 70}%

Recommended Response: ${formData.sender.includes('suspicious') ? 'Mark as spam and delete' : 'Professional acknowledgment and follow-up'}`;
        
        setResponse(mockResponse);
      }
    } catch (error) {
      console.error('Error submitting email:', error);
      setResponse('Error processing email. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-100 flex">
      {/* Sidebar */}
      <div className="w-80 bg-white shadow-lg p-6">
        <div className="flex items-center mb-8">
          <div className="w-12 h-12 bg-green-600 rounded-lg flex items-center justify-center mr-3">
            <svg className="w-8 h-8 text-white" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 2L2 7v10c0 5.55 3.84 9.95 9 11 5.16-1.05 9-5.45 9-11V7l-10-5z"/>
              <path d="M9 12l2 2 4-4" stroke="white" strokeWidth="2" fill="none"/>
            </svg>
          </div>
          <div>
            <h1 className="text-xl font-bold text-gray-800">BNP PARIBAS</h1>
            <p className="text-sm text-gray-600 font-semibold">ASSET MANAGEMENT</p>
          </div>
        </div>

        <div className="mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4">Alfred Email Butler</h2>
        </div>

        <div className="border-t border-gray-200 pt-6 mb-8">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">About</h3>
          <p className="text-gray-600 text-sm leading-relaxed">
            Alfred is your personal email butler, powered by AI to help process, classify and respond to emails efficiently
          </p>
        </div>

        <div className="border-t border-gray-200 pt-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Demo Emails</h3>
          <div className="space-y-3">
            <button
              onClick={loadSampleBusinessInquiry}
              className="w-full px-4 py-3 bg-gray-50 hover:bg-gray-100 border border-gray-300 rounded-lg text-left text-sm font-medium text-gray-700 transition-colors"
            >
              Load Sample Business Inquiry
            </button>
            <button
              onClick={loadSampleSpamEmail}
              className="w-full px-4 py-3 bg-gray-50 hover:bg-gray-100 border border-gray-300 rounded-lg text-left text-sm font-medium text-gray-700 transition-colors"
            >
              Load Sample Spam Email
            </button>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 p-8">
        <div className="max-w-4xl mx-auto">
          <h1 className="text-3xl font-bold text-gray-800 mb-8">Compose or Load an Email</h1>
          
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="space-y-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Sender
                </label>
                <input
                  type="email"
                  name="sender"
                  value={formData.sender}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent outline-none transition-all"
                  placeholder="Enter sender email address"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Subject
                </label>
                <input
                  type="text"
                  name="subject"
                  value={formData.subject}
                  onChange={handleInputChange}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent outline-none transition-all"
                  placeholder="Enter email subject"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Email Body
                </label>
                <textarea
                  name="emailBody"
                  value={formData.emailBody}
                  onChange={handleInputChange}
                  rows={12}
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-transparent outline-none transition-all resize-none"
                  placeholder="Enter email content"
                />
              </div>

              <div className="flex justify-start">
                <button
                  onClick={submitEmail}
                  disabled={isLoading}
                  className="px-6 py-3 bg-green-600 hover:bg-green-700 disabled:bg-gray-400 text-white font-medium rounded-lg transition-colors flex items-center space-x-2"
                >
                  {isLoading && (
                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                    </svg>
                  )}
                  <span>{isLoading ? 'Processing...' : 'Submit Email'}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Response Display */}
          {response && (
            <div className="mt-8 bg-white rounded-lg shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-4">Analysis Results</h2>
              <div className="bg-gray-50 rounded-lg p-4">
                <pre className="whitespace-pre-wrap text-sm text-gray-700 font-mono">
                  {response}
                </pre>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}



<!DOCTYPE html>
<html>
<head>
  <title>Alfred Email Butler</title>
  <link rel="stylesheet" href="/static/style.css">
</head>
<body>
  <div class="app-container">
    <div class="sidebar">
      <h1>BNP PARIBAS<br>ASSET MANAGEMENT</h1>
      <p>Alfred is your personal email butler, powered by AI.</p>
      <div class="demo-buttons">
        <button onclick="loadSample('business')">Load Sample Business Inquiry</button>
        <button onclick="loadSample('spam')">Load Sample Spam Email</button>
      </div>
    </div>
    <div class="main">
      <h2>Compose or Load an Email</h2>
      <form method="post" action="/submit">
        <input type="text" name="sender" id="sender" placeholder="Sender" value="{{ sender | default('') }}" required>
        <input type="text" name="subject" id="subject" placeholder="Subject" value="{{ subject | default('') }}" required>
        <textarea name="body" id="body" placeholder="Email Body" required>{{ body | default('') }}</textarea>
        <button type="submit">Submit Email</button>
      </form>

      {% if response %}
      <div class="response-box">{{ response }}</div>
      {% endif %}
    </div>
  </div>

  <script>
    const samples = {
      business: {
        sender: "client@example.com",
        subject: "Business Inquiry",
        body: "We are interested in your services. Can you provide more information?"
      },
      spam: {
        sender: "spam@malicious.com",
        subject: "You won a prize!",
        body: "Click this link to claim your prize now."
      }
    };

    function loadSample(type) {
      const sample = samples[type];
      document.getElementById('sender').value = sample.sender;
      document.getElementById('subject').value = sample.subject;
      document.getElementById('body').value = sample.body;
    }
  </script>
</body>
</html>


/* src/App.css */

.app-container {
  display: flex;
  height: 100vh;
  background-color: #f4f6f8;
  font-family: 'Segoe UI', sans-serif;
  color: #1a1a1a;
}

.sidebar {
  width: 280px;
  background-color: white;
  border-right: 1px solid #d1d1d1;
  padding: 30px 20px;
}

.sidebar h1 {
  font-size: 20px;
  margin-bottom: 12px;
}

.sidebar p {
  font-size: 14px;
  margin-bottom: 40px;
  color: #333;
}

.sidebar .demo-buttons button {
  display: block;
  width: 100%;
  margin-bottom: 12px;
  padding: 10px;
  font-size: 14px;
  background-color: #00754a;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
}

.sidebar .demo-buttons button:hover {
  background-color: #005f3a;
}

.main {
  flex: 1;
  padding: 40px 60px;
}

h2 {
  font-size: 22px;
  margin-bottom: 24px;
}

form {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

input,
textarea {
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
}

textarea {
  resize: vertical;
  min-height: 100px;
}

button[type="submit"] {
  align-self: flex-start;
  padding: 10px 16px;
  background-color: #00754a;
  color: white;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  margin-top: 10px;
}

button[type="submit"]:hover {
  background-color: #005f3a;
}

.response-box {
  margin-top: 20px;
  padding: 12px;
  background-color: #eef6f0;
  border-left: 4px solid #00754a;
  border-radius: 4px;
  font-size: 14px;
}
    
    
