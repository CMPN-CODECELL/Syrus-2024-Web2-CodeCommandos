import React, { useEffect, useState } from "react";
import styled from "styled-components";
import axios from "axios";
import { getTrip } from "../../utils/api-routes";
import { Link } from "react-router-dom";
export default function ShowTrip({ setAddShowTrip }) {
  const [trips,setTrips] = useState([])
  const user = JSON.parse(localStorage.getItem("user"));
  useEffect(() => {
    async function getTrips() {
      try {
        const { data } = await axios.get(getTrip, {
          headers: {
            Authorization: user.token,
          },
        });
        setTrips(data)
        console.log(data);
      } catch (error) {
        console.log(error.message);
      }
    }
    getTrips();
  }, []);
  const handleAIButtonClick = () => {
    // Redirect to the AI link
    window.location.href = "https://travel-email-sender-7qklh6qvjevt2nkysgs7rj.streamlit.app/";
  };
  return (
    <Container>
      <div className="row">
        <div className="mytrips">
          {
            trips && 
            trips.map((item)=>(
              <div className={`${item ? 'trip' : 'notrip'}`}>
                <h4>{item.departureDate}</h4>
                <h4>{item.returnDate}</h4>
                <h4>{item.arrivalCity}</h4>
                <img src={`http://localhost:5000${item.ticket}`} alt="" />
                {
                  item.tripMates.map((item)=>(
                    <h4>{item}</h4>
                  ))
                }
              </div>
            ))
          }
        </div>
        <div className="myposts"></div>
      </div>
      <FloatingActionButton2
        onClick={handleAIButtonClick}
      >
        AI
      </FloatingActionButton2>
      <FloatingActionButton onClick={() => setAddShowTrip("add")}>
        +
      </FloatingActionButton>
    </Container>
  );
}
const Container = styled.div`
  height: 100%;
  width: 100%;
  .row {
    height: 100%;
    display: grid;
    grid-template-columns: 50% 50%;
    .mytrips {
      height: 100%;
      width: 100%;
      .trip{
        background-color: #282a34;
        border-radius: 2rem;
        padding: 1rem;
        h4{
          color: white;
        }

      }
      .notrip{
        display: none;
      }
    }
    .myposts {
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
