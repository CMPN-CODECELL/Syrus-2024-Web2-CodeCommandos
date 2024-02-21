import React, { useEffect, useState } from "react";
import axios from "axios";
import styled from "styled-components";
import {FaMapMarked as Map} from 'react-icons/fa'
import DestDetails from "./Feed/DestDetails";
import { getDestinRoute } from "../utils/api-routes";
import AddTrip from "./Feed/AddTrip";
import ShowTrip from "./Feed/ShowTrip";
import Search from "./Feed/Search";
import AboutUs from "./Feed/AboutUs";
export default function Feed({feed}) {
  const [data, setData] = useState([]);
  const [addShowTrip,setAddShowTrip] = useState("show")
  const handleMap = ()=>{
    window.location.href = 'http://codecommandos-mapintegration.netlify.app/'
  }
  useEffect(() => {
    async function getDest() {
      try {
        const { token } = JSON.parse(localStorage.getItem("user"));
        const { data } = await axios.get(getDestinRoute, {
          headers: {
            Authorization: token,
          },
        });
        setData(data);
      } catch (error) {
        console.log(error.message);
      }
    }
    getDest();
  }, []);
  switch (feed) {
    case "home":
      return (
        <Container>
          <DestDetailsContainer>
            {data.map((item, index) => (
              <DestDetails key={index} eventData={item} />
            ))}
          </DestDetailsContainer>
          <FloatingActionButton onClick={handleMap}>
            <Map/>
          </FloatingActionButton>
        </Container>
      );

    case "trip":
      return (
        <Container>
          {addShowTrip==="add" ? (
            <AddTripContainer>
              <AddTrip setAddShowTrip={setAddShowTrip}/>
            </AddTripContainer>
          ) : (
            <TripDetailsContainer>
              <ShowTrip setAddShowTrip={setAddShowTrip}/>
            </TripDetailsContainer>
          )}
        </Container>
      );
      case "search":
        return(
          <Container>
            <Search/>
          </Container>
        )
        }
}

const Container = styled.div`
  height: 100vh;
  width: 100%;
  background-color: #3c3f51;
  border-radius: 3rem;
  align-items: center;
  padding: 2rem;
  overflow: auto;
`;

const DestDetailsContainer = styled.div`
  gap: 1rem;
  user-select: none;
  overflow-y: auto;
  &::-webkit-scrollbar {
    width: 0.5rem;
    &-thumb {
      background-color: #ffffff39;
      width: 0.1rem;
      border-radius: 1rem;
    }
  }
  height: 90vh;
  display: flex;
  flex-direction: column;
  border-radius: 4rem;
  flex-wrap: nowrap;
`;
const AddTripContainer = styled.div``;
const TripDetailsContainer = styled.div``;
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