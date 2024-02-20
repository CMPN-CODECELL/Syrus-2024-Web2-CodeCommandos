import React, { useState, useEffect } from "react";
import axios from "axios";
import styled from "styled-components";
import Logo from "../../assets/logo.png";
import Back from "./Back";
import { ToastContainer, toast } from "react-toastify";
import { FaCheckCircle as Accept } from "react-icons/fa";
import { AiOutlineCloseCircle as Reject } from "react-icons/ai";
import "react-toastify/dist/ReactToastify.css";
import { acceptTripRoute, getTrip, rejectTripRoute } from "../../utils/api-routes";
export default function ({ changeMenu, setMessages }) {
  const [trips, setTrips] = useState([]);
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
      } catch (error) {
        console.log(error.message);
      }
    }
    getTrips();
  }, []);
  const addTrip = async(trip)=>{
    try {
      const {data} = await axios.post(acceptTripRoute,{trip},{headers: {
        Authorization: user.token
      }})
      console.log(data);
    } catch (error) {
      console.log(error.message);
    }
  }
  const rejectTrip = async(trip)=>{
    try {
      const {data} = await axios.post(rejectTripRoute,{trip},{headers:{
        Authorization: user.token
      }})
      console.log(data);
      setTrips(data)
    } catch (error) {
      console.log(error.message);
    }
  }
  return (
    <>
      <Container>
        <div className="top-bar">
          <div className="back">
            <Back changeMenu={changeMenu} />
          </div>
        </div>
        <h4>Trips</h4>
        <div className="users">
          {trips.map((item, index) => (
            <div
              key={index}
              className="user"
              onClick={() => {
                  changeMenu("Messages");
                  setMessages([true, item]);
              }}
            >
              <img src={Logo} alt="userphoto" key={index} />
              <div
                className={`details "}`}
                key={index}
              >
                <h4>{`Trip Id: ${item._id}`}</h4>
                <div className="butt">
                  <h4>{`To: ${item.arrivalCity}`}</h4>
                </div>
              </div>
            </div>
          ))}
        </div>
      </Container>
      <ToastContainer />
    </>
  );
}

const Container = styled.div`
  height: 100vh;
  width: 100%;
  display: flex;
  flex-direction: column;
  padding: 1rem;
  gap: 0.2rem;
  background-color: #282a34;
  h4 {
    color: white;
  }
  .top-bar {
    display: flex;
    flex-direction: row;
    align-items: center;
    gap: 1rem;
    padding-bottom: 1rem;
    .search {
      border-radius: 0.5rem;
      background-color: #373842;
      display: flex;
      flex-direction: row;
      justify-content: center;
      gap: 1rem;
      width: fit-content;
      input {
        outline: none;
        padding: 0.5rem;
        background-color: transparent;
        color: #7e7f89;
        border: none;
      }
      .icon {
        align-self: center;
        color: #595a63;
      }
    }
  }
  .users {
    width: 100%;
    padding: 1rem 0 0 0;
    display: flex;
    flex-direction: column;
    gap: 1rem;
    overflow: auto;
    &::-webkit-scrollbar {
      width: 0.2rem;
      &-thumb {
        background-color: #ffffff39;
        width: 0.1rem;
        border-radius: 1rem;
      }
    }
    .user {
      cursor: pointer;
      user-select: none;
      display: flex;
      flex-direction: row;
      gap: 1rem;
      img {
        height: 5rem;
        width: 5rem;
        border-radius: 2.5rem;
        padding: 0%.1rem;
        border: 0.1rem solid #4e3eff;
      }
      .details {
        overflow-x: hidden;
        align-self: center;
        h4 {
          overflow: hidden;
          color: white;
        }
        .butt {
          display: flex;
          flex-direction: row;
          gap: 1rem;
        }
      }
      .nobutton {
        .butt {
          display: none;
        }
      }
    }
  }
`;
const Button = styled.button`
  display: flex;
  justify-content: center;
  align-items: center;
  padding: 0.5rem;
  border-radius: 0.5rem;
  border: none;
  cursor: pointer;
  background-color: #9186f3;
  svg {
    font-size: 0.8rem;
  }
`;
