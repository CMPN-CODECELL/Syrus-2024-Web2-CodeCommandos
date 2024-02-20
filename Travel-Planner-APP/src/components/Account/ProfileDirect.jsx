import React, { useState,useEffect } from "react";
import axios from 'axios'
import styled from "styled-components";
import {FaCheckCircle as Accept} from 'react-icons/fa'
import {AiOutlineCloseCircle as Reject} from "react-icons/ai";
import {FaUserPlus as Add} from "react-icons/fa";
import { FaSearch as Search } from "react-icons/fa";
import Logo from "../../assets/logo.png";
import Back from "./Back";
import { ToastContainer, toast } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";
import { addFriendRoute, addRequestRoute, getUsers } from "../../utils/api-routes";
export default function ({ changeMenu,setMessages}) {
  const [users,setUsers] = useState([])
  const user = JSON.parse(localStorage.getItem('user'))
  useEffect(()=>{
    async function friends(){
      try {
        const {data} = await axios.get(getUsers,{headers:{
          Authorization: user.token
        }})
        setUsers(data)
      } catch (error) {
        console.log(error.message);
      }
    }
    friends()
  },[])

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
        {users.map((item,index) => (
          <div key={index}
            className="user"
            onClick={() => {
              changeMenu("Messages");
              setMessages(true);
            }}
          >
            <img src={Logo} alt="userphoto" key={index}/>
            <div className="details" key={index}>
              <h4>{item.name}</h4>
            </div>
          </div>
        ))}
      </div>
    </Container>
    <ToastContainer/>
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
  h4{
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
        align-self: center;
        h4 {
          color: white;
        }
        .buttons{
          display: flex;
          flex-direction: row;
          gap: 1rem;
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
