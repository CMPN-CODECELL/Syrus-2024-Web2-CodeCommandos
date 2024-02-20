import React, { useState } from "react";
import ReactModal from "react-modal";
import styled from "styled-components";
import LikePost from "./LikePost";
import CommentPost from "./CommentPost";
import SharePost from "./SharePost";
export default function Dialog({ modalOpen, close, modalType }) {
  const customStyles = {
    content: {
      height: "100%",
      width: "100%",
      top: "50%",
      left: "50%",
      transform: "translate(-50%, -50%)",
      margin: 0,
      padding: 0,
      border: 'none',
      backgroundColor: '#282a34',
      zIndex: '9'
    },
    overlay: {
      paddingBottom: "1rem",
      height: "100%",
      width: "100%",
      position: "absolute",
      top: 0,
      right: 0,
      bottom: 0,
      left: 0,
      backgroundColor: "rgba(0, 0, 0, 0)",
    },
  };
  return (
    <Container>
      <ReactModal
        isOpen={modalOpen}
        style={customStyles}
        onRequestClose={close}
        parentSelector={() => document.getElementById("addpost")}
        ariaHideApp={false}
      >
        <div className="top"></div>
        <div className="section">
          {modalType === "Likes" && <LikePost />}
          {modalType === "Comments" && <CommentPost />}
          {modalType === "Shares" && <SharePost />}
        </div>
      </ReactModal>
    </Container>
  );
}
const Container = styled.div`
padding: 0 1rem;
`;
