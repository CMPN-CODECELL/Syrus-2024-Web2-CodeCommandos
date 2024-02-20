import React, { useEffect, useState } from "react";
import styled from "styled-components";
import Register from "../components/Auth/Register";
import Login from "../components/Auth/Login";
import { useNavigate } from "react-router-dom";
export default function Auth() {
  const navigate = useNavigate()
const [auth,setAuth] = useState('Register')
const changeAuth = (auth)=>{
    setAuth(auth)
}
useEffect(()=>{
  const user = localStorage.getItem('user')
  if(user){
    navigate('/')
  }
})
  return (
    <Container>
      {
        auth === "Register" ? <Register changeAuth={changeAuth}/> : <Login changeAuth={changeAuth}/> 
      }
      <Register />
    </Container>
  );
}
const Container = styled.div`
  height: 100vh;
  width: 100vw;
  display: grid;
  align-items: center;
  background-color: #282a34;
`;
