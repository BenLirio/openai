#use "topfind"
#require "unix"
open Unix
let rc, wc = open_connection (ADDR_INET ((inet_addr_of_string "0.0.0.0"), 80))
let get_req = Bytes.of_string "GET / HTTP/1.1\r\n"

let () =
    output wc get_req 0 (Bytes.length get_req);
    flush wc;
    Printf.printf "%s\n" (input_line rc)
