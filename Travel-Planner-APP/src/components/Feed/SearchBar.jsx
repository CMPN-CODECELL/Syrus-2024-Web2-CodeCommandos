import React, { useState } from "react";
import styled from "styled-components";
import { FaSearch as Search } from "react-icons/fa";
export default function SearchBar() {
  const [text, setText] = useState("");
  return (
    <Container>
      <div className="search">
        <Search className="icon" />
        <input
          type="text"
          placeholder="Search"
          value={text}
          onChange={(e) => setText(e.target.value)}
        />
      </div>
    </Container>
  );
}
const Container = styled.div`
  padding: 0.8rem 2rem 0.8rem 2rem;
  .search {
    padding: 0.5rem;
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
`;
