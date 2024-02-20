import React, { useState } from "react";
import styled from "styled-components";
import Hi from "../../assets/download.jpeg";
import { FaSearch as Search } from "react-icons/fa";
export default function PostDirect() {
  const arr = [1,1,2,2];
  const [text, setText] = useState("");
  return (
    <Container>
      <div className="users">
        <div className="top">
          <div className="search">
            <Search className="icon" />
            <input
              type="text"
              placeholder="Search"
              value={text}
              onChange={(e) => setText(e.target.value)}
            />
          </div>
        </div>
        <div className="user">
          {arr.map(() => (
            <div className="column">
              <img src={Hi} alt="" />
              <h6>username</h6>
            </div>
          ))}
        </div>
        <div className="button">
          <button>Send</button>
        </div>
      </div>
    </Container>
  );
}
const Container = styled.div`
  height: 100%;
  width: 100%;
  .users {
    display: grid;
    grid-template-rows: 25% 65% 10%;
    .top {
      height: 100%;
      padding: 0.3rem;
      display: flex;
      flex-direction: row;
      gap: 1rem;
      h3 {
        color: white;
      }
      .search {
        height: 100%;
        width: 100%;
        border-radius: 0.5rem;
        background-color: #373842;
        display: flex;
        flex-direction: row;
        justify-content: center;
        gap: 1rem;
        input {
          outline: none;
          padding: 0.1rem;
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
    .user {
      height: 100%;
      padding: 0.3rem;
      display: grid;
      grid-template-columns: 33% 33% 33%;
      overflow-y: auto;
      &::-webkit-scrollbar {
        width: 0.2rem;
        &-thumb {
          background-color: #ffffff39;
          width: 0.1rem;
          border-radius: 1rem;
        }
      }
      .column {
        height: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        img {
          height: 3rem;
          width: 3rem;
          padding: 0.1rem;
          border-radius: 1.5rem;
          border: 0.1rem solid #4e3eff;
        }
        h6 {
          color: white;
        }
      }
    }
    .button {
      margin: 0.5rem;
      padding: 0.3rem;
      height: 100%;
      button {
        align-self: center;
        background-color: #997af0;
        color: white;
        width: 100%;
        padding: 0.3rem;
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
    }
  }
`;
