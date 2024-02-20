import React, { useRef, useState } from "react";
import styled from "styled-components";
import App from "../../assets/logo.png";
import Logo from "../../assets/logo.png";
import { Carousel } from "react-responsive-carousel";
import "react-responsive-carousel/lib/styles/carousel.min.css";
import {
  FaHeart as Like,
  FaComment as Comment,
  FaShare as Share,
} from "react-icons/fa";
import Dialog from "./Dialog";
export default function Post({addpost,postData}) {
  const [seeMore, setSeeMore] = useState(false);
  const [selectedButton, setSelectedButton] = useState("");
  const arr = [1, 1, 1, 17];
  const handleCloseModal = () => {
    setModalOpen([false, ""]);
  };
  return (
    <Container>
      <div className="body">
        <div className="side1">
          <div className="top">
            <div className="detail" >
              <img src={App} alt="" />
              <h3>username</h3>
            </div>
            <div className="side2">
              <div className="buttons">
                <div className="like btn">
                  <Button
                    onClick={() => {
                      setSelectedButton("Like");
                    }}
                  >
                    <Like
                      className="icon"
                      size="25"
                      style={{
                        color: `${selectedButton === "Like" ? "red" : "white"}`,
                      }}
                    />
                  </Button>
                  <h6 onClick={() => setModalOpen([true, "Likes"])}>196</h6>
                </div>
                <div
                  className="comment btn"
                  onClick={() => {
                    setSelectedButton("Comment");
                    setModalOpen([true, "Comments"]);
                  }}
                >
                  <Button>
                    <Comment
                      className="icon"
                      size="25"
                      style={{
                        color: `${
                          selectedButton === "Comment" ? "red" : "white"
                        }`,
                      }}
                    />
                  </Button>
                  <h6>567</h6>
                </div>
                <div
                  className="share btn"
                  onClick={() => {
                    setSelectedButton("Share");
                    setModalOpen([true, "Shares"]);
                  }}
                >
                  <Button>
                    <Share
                      className="icon"
                      size="25"
                      style={{
                        color: `${
                          selectedButton === "Share" ? "red" : "white"
                        }`,
                      }}
                    />
                  </Button>
                  <h6>89</h6>
                </div>
              </div>
            </div>
          </div>
          <div className="caption">
          <p style={{ height: !seeMore ? "1.3em" : "auto", overflow: "hidden" }}>
              rtdyfgmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmjhyzxg{""}
              </p>
          </div>
          <div className="image">
            <Carousel className="slider">
              {arr.map((_, index) => (
                <img src={Logo} alt="" key={index} />
              ))}
            </Carousel>
          </div>
        </div>
      </div>
    </Container>
  );
}
const Container = styled.div`
  width: 100%;
  height: 100%;
  user-select: none;
  .body {
    height: 100%;
    width: 100%;
    background-color: #3c3f51;
    border-radius: 0.5rem;
    border: 0.1rem solid #4e3eff;
    .side1 {
      display: flex;
      flex-direction: column;
      .caption {
        user-select: text;
        padding: 0.4rem;
        border-bottom: 0.1rem solid #4e3eff;
        white-space: normal;
        overflow: hidden;
        word-wrap: break-word;
        p {
          line-height: 1.3em;
          font-size: medium;
          color: white;
        }
      }
      .top {
        display: flex;
        justify-content: space-between;
        border-bottom: 0.1rem solid #4e3eff;
        .detail {
          display: flex;
          flex-direction: row;
          gap: 1rem;
          h3 {
            color: white;
            align-self: center;
          }
          img {
            align-self: center;
            margin: 0.3rem;
            height: 3rem;
            width: 3rem;
            border-radius: 1.5rem;
            padding: 0.1rem;
          }
        }
        .side2 {
          align-self: center;
          padding: 0.2rem;
          .buttons {
            display: flex;
            gap: 2rem;
            align-items: center;
            justify-content: center;
            .btn {
              display: flex;
              flex-direction: column;
              align-items: center;
              gap: 0.4rem;
              h6 {
                color: white;
              }
            }
          }
        }
      }
      .image {
        .carousel-root {
          .carousel {
            width: fit-content;
            .thumbs-wrapper {
              display: none;
            }
          }
        }
        .slider {
          height: 29rem;
          width: 100%;
          img {
            max-width: 100%;
            @media screen and (min-width: 1400px) {
              height: 26rem;
            }
            border-bottom-left-radius: 0.5rem;
          }
        }
      }
    }
  }
`;
const Button = styled.button`
  border: none;
  color: white;
  background-color: transparent;
`;
