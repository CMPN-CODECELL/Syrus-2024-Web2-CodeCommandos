import React, { useEffect } from "react";
import styled from "styled-components";
import axios from "axios";
import { getTrip } from "../../utils/api-routes";
export default function ShowTrip({ setAddShowTrip }) {
  const user = JSON.parse(localStorage.getItem('user'))
  useEffect(()=>{
    async function getTrips(){
      try {
        const{data} = await axios.get(getTrip,{headers:{
          Authorization: user.token
        }})
        console.log(data);
      } catch (error) {
        console.log(error.message);
      }
    }
    getTrips()
  },[])
  return (
    <Container>
      <div className="row">
        <div className="mytrips">
       
        </div>
        <div className="myposts">
          
        </div>
      </div>
      <FloatingActionButton2>AI</FloatingActionButton2>
      <FloatingActionButton onClick={() => setAddShowTrip("add")}>
        +
      </FloatingActionButton>
    </Container>
  );
}
const Container = styled.div`
  height: 100%;
  width: 100%;
  .row{
    height: 100%;
    display: grid;
    grid-template-columns: 50% 50%;
    .mytrips{
      flex: 1;
      height: 100%;
      width: 100%;
    }
    .myposts{
      height: 100%;
      width: 100%;
      overflow: auto;
      &-thumb {
      background-color: #ffffff39;
      width: 0.1rem;
      border-radius: 1rem;
    }
    }
  }
`;
const FloatingActionButton = styled.button`
  position: fixed;
  bottom: 20px;
  right: 20px;
  background-color: #61dafb;
  color: white;
  padding: 15px;
  border: none;
  font-size: 1.5em;
  border-radius: 50%;
  cursor: pointer;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  transition: background-color 0.3s ease-in-out;

  &:hover {
    background-color: #4e0eff;
  }
`;
const FloatingActionButton2 = styled.button`
  position: fixed;
  bottom: 80px;
  right: 20px;
  background-color: #61dafb;
  color: white;
  padding: 15px;
  border: none;
  font-size: 1.5em;
  border-radius: 50%;
  cursor: pointer;
  box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
  transition: background-color 0.3s ease-in-out;

  &:hover {
    background-color: #4e0eff;
  }
`;
