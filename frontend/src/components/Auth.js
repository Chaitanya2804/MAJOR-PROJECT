// frontend/src/components/Auth.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

function Auth({ setUser, darkMode }) {
  const [isLogin, setIsLogin] = useState(true);
  const [formData, setFormData] = useState({
    name: '',
    email: '',
    password: ''
  });
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    
    try {
      const endpoint = isLogin ? '/api/auth/login' : '/api/auth/register';
      const data = isLogin ? 
        { email: formData.email, password: formData.password } :
        formData;
      
      const response = await axios.post(`http://localhost:5000${endpoint}`, data);
      
      localStorage.setItem('token', response.data.token);
      localStorage.setItem('user', JSON.stringify(response.data.user));
      setUser(response.data.user);
      navigate('/');
    } catch (err) {
      setError(err.response?.data?.msg || 'Something went wrong');
    }
  };

  return (
    <div className="auth-container">
      <div className="auth-card">
        <div className="auth-header">
          <h2>🏥 VITAS-AI</h2>
          <h3>{isLogin ? 'Login' : 'Sign Up'}</h3>
        </div>
        
        {error && <div className="error-msg">{error}</div>}
        
        <form onSubmit={handleSubmit}>
          {!isLogin && (
            <input
              type="text"
              name="name"
              placeholder="Full Name"
              value={formData.name}
              onChange={handleChange}
              required={!isLogin}
            />
          )}
          
          <input
            type="email"
            name="email"
            placeholder="Email"
            value={formData.email}
            onChange={handleChange}
            required
          />
          
          <input
            type="password"
            name="password"
            placeholder="Password"
            value={formData.password}
            onChange={handleChange}
            required
          />
          
          <button type="submit" className="btn-primary">
            {isLogin ? 'Login' : 'Sign Up'}
          </button>
        </form>
        
        <p className="auth-switch">
          {isLogin ? "Don't have an account? " : "Already have an account? "}
          <button 
            type="button"
            onClick={() => setIsLogin(!isLogin)}
            className="link-btn"
          >
            {isLogin ? 'Sign Up' : 'Login'}
          </button>
        </p>
        
        <button 
          onClick={() => navigate('/')}
          className="btn-secondary"
        >
          Back to Home
        </button>
      </div>
    </div>
  );
}

export default Auth;