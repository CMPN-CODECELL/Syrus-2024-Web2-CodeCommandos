import React, { useState, useEffect, useRef } from "react";
import styled from "styled-components";
import { v4 as uuidv4 } from "uuid";
import axios from "axios";
import Back from "./Back";
import Logo from "../../assets/logo.png";
import DM from "./DM";
export default function Messages({changeMenu,setMessages}) {
  const [messages, setMessage] = useState([]);
  const scrollRef = useRef();
  const [arrivalMessage, setArrivalMessage] = useState(null);
  const arr = [1, 1];
  return (
    <Container>
      <div className="chat-header">
        <div className="back">
          <Back changeMenu={changeMenu} setMessages={setMessages}/>
        </div>
        <div className="user-details">
          <div className="avatar">
            <img src={Logo} alt="" />
          </div>
          <div className="username">
            <h3>username</h3>
          </div>
        </div>
      </div>
      <div className="chat-messages">
        {arr.map((message) => {
          return (
            <div ref={scrollRef} key={uuidv4()}>
              <div
                className={`message ${message == 1 ? "sended" : "recieved"}`}
              >
                <div className="content ">
                  <p>{message}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>
      <div className="dm">
        <DM />
      </div>
    </Container>
  );
}

const Container = styled.div`
  height: 100vh;
  display: grid;
  grid-template-rows: 10% 80% 10%;
  gap: 0.1rem;
  overflow: hidden;
  @media screen and (min-width: 720px) and (max-width: 1080px) {
    grid-template-rows: 15% 70% 15%;
  }
  .chat-header {
    display: flex;
    gap: 1rem;
    justify-content: flex-start;
    align-items: center;
    padding: 0.5rem;
    border-bottom: oldlace.1rem solid #3c3f51;
    .user-details {
      display: flex;
      align-items: center;
      gap: 1rem;
      .avatar {
        img {
          height: 3rem;
          width: 3rem;
          border-radius: 1.5rem;
          padding: 0.1rem;
          border: 0.1rem solid #4e0eff;
        }
      }
      .username {
        h3 {
          color: white;
        }
      }
    }
  }
  .chat-messages {
    padding: 1rem 2rem;
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
    .message {
      display: flex;
      align-items: center;
      .content {
        max-width: 40%;
        overflow-wrap: break-word;
        padding: 1rem;
        font-size: 1.1rem;
        border-radius: 1rem;
        color: #d1d1d1;
        @media screen and (min-width: 720px) and (max-width: 1080px) {
          max-width: 70%;
        }
      }
    }
    .sended {
      justify-content: flex-end;
      .content {
        background-color: #4f04ff21;
      }
    }
    .recieved {
      justify-content: flex-start;
      .content {
        background-color: #9900ff20;
      }
    }
  }
  .dm {
    margin: 0.5rem;
    height: 100%;
  }
`;
