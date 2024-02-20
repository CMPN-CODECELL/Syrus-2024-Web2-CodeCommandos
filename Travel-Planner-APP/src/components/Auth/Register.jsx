import React, { useState } from "react";
import styled from "styled-components";
import { ToastContainer, toast } from "react-toastify";
import axios from "axios";
import signup from "../../assets/signup.gif";
import "react-toastify/dist/ReactToastify.css";
import { registerRoute } from "../../utils/api-routes";
import { useNavigate } from "react-router-dom";
export default function Register({ changeAuth }) {
  const navigate = useNavigate();
  const [values, setValues] = useState({
    name: "",
    username: "",
    email: "",
    password: "",
  });
  const handleSubmit = async (event) => {
    event.preventDefault();
    if (handleValidation()) {
      const { name, username, email, password } = values;
      try {
        const { data } = await axios.post(registerRoute, {
          name,
          username,
          email,
          password,
        });
        localStorage.setItem("user", JSON.stringify(data));
        navigate("/");
      } catch (error) {
        toast.error(`Error registering`, toastOptions);
      }
    }
  };
  const handleChange = (event) => {
    setValues({ ...values, [event.target.name]: event.target.value });
  };
  const toastOptions = {
    position: "bottom-right",
    autoClose: 8000,
    pauseOnHover: true,
    draggable: true,
    theme: "dark",
  };
  const handleValidation = () => {
    const { username, email, password } = values;
    if (username.length < 3) {
      toast.error("Username should be greater than 3 characters", toastOptions);
    } else if (password.length < 7) {
      toast.error("Password should be greater than 7 characters", toastOptions);
      return false;
    } else if (email === "") {
      toast.error("Email should be filled.");
      return false;
    }
    return true;
  };
  return (
    <>
      <FormContainer>
        <div className="row">
          <form onSubmit={(e) => handleSubmit(e)}>
            <div className="brand">
              <h1>Travel Planner App</h1>
            </div>
            <input
              type="name"
              placeholder="Name"
              name="name"
              onChange={(e) => handleChange(e)}
            />
            <input
              type="text"
              placeholder="UserName"
              name="username"
              onChange={(e) => handleChange(e)}
            />
            <input
              type="text"
              placeholder="Email"
              name="email"
              onChange={(e) => handleChange(e)}
            />
            <input
              type="password"
              placeholder="Password"
              name="password"
              onChange={(e) => handleChange(e)}
            />

            <button type="submit">Create User</button>
            <span>
              Already have an account?{" "}
              <span onClick={() => changeAuth("Login")}>Login</span>
            </span>
          </form>
          <img src={signup} alt="sdvfv" />
        </div>
      </FormContainer>
      <ToastContainer />
    </>
  );
}
const FormContainer = styled.div`
  height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  justify-content: center;
  gap: 1rem;
  align-items: center;
  background-color: #282a34;
  .row {
    display: flex;
    .brand {
      display: flex;
      align-items: center;
      justify-content: center;
      img {
        height: 5rem;
      }
      h1 {
        color: white;
        text-transform: uppercase;
      }
    }
    form {
      border: 0.2rem solid #4e0eff;
      display: flex;
      flex-direction: column;
      gap: 2rem;
      background-color: #282a34;
      padding: 3rem 5rem;
      input {
        background-color: transparent;
        padding: 1rem;
        border: 0.1rem solid #4e0eff;
        border-radius: 0.4rem;
        color: white;
        width: 100%;
        font-size: 1rem;
        &:focus {
          border: 0.1rem solid #997af0;
          outline: none;
        }
      }
      button {
        background-color: #997af0;
        color: white;
        width: 100%;
        padding: 1rem 2rem;
        border: none;
        font-weight: bold;
        cursor: pointer;
        border-radius: 0.4rem;
        font-size: 1rem;
        text-transform: uppercase;
        transition: 0.5s ease-in-out;
        &:hover {
          background-color: #4e0eff;
        }
      }
      span {
        text-transform: uppercase;
        color: white;
        span {
          cursor: pointer;
          color: #4e0eff;
          text-decoration: none;
          font-weight: bold;
        }
      }
    }
  }
`;
