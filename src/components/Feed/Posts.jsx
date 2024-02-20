import React, { useRef, useState } from "react";
import styled from "styled-components";
import Post from "./Post";
import Back from "../Account/Back";
export default function Posts({ inUser, changeMenu }) {
  const arr = [
    1,5,5,5];
  return (
    <Container>
      <div className="posts">
        {inUser && <Back changeMenu={changeMenu} />}
        <div className="post">
          {arr.map((_, index) => (
            <Post
              arr={arr}
              className="slider"
              changeMenu={changeMenu}
              key={index}
            />
          ))}
        </div>
      </div>
    </Container>
  );
}
const Container = styled.div`
  width: 100%;
  height: 100%;
  padding: 0 1rem;
  margin-bottom: 1rem;
  overflow-y: auto;
  &::-webkit-scrollbar {
        width: 0.2rem;
        height: 0.2rem;
        &-thumb {
          background-color: #ffffff39;
          width: 0.1rem;
          border-radius: 1rem;
        }
      }
  .posts {
    height: 100%;
    width: 100%;
    padding-bottom: 1rem;
    align-items: stretch;
    .post {
      height: 100%;
      background-color: #282a34;
      width: 100%;
      padding: 0.5rem;
      display: flex;
      flex-direction: column;
      gap: 0.5rem;
      border-radius: 0.5rem;
      overflow-y: auto;
      &::-webkit-scrollbar {
        width: 0.2rem;
        height: 0.2rem;
        &-thumb {
          background-color: #ffffff39;
          width: 0.1rem;
          border-radius: 1rem;
        }
      }
    }
  }
`;
