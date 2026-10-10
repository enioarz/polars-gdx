/*
* GAMS - General Algebraic Modeling System GDX API
*
* Copyright (c) 2017-2026 GAMS Software GmbH <support@gams.com>
* Copyright (c) 2017-2026 GAMS Development Corp. <support@gams.com>
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include "gmsstrm.hpp"
#include "../doctest.hpp"
#include <fstream>
#include <filesystem>
#include <algorithm>

using namespace std::literals::string_literals;
using namespace gdlib::gmsstrm;

namespace tests::gdlibtests::gmsstrmtests
{

TEST_SUITE_BEGIN( "gdlib::gmsstrm" );

template<typename StrmClazz>
void testWritingOneByteIntoTextFile()
{
   const uint8_t examplePayload { 23 };
   const std::string exampleFn { "test.txt"s };

   if( std::filesystem::exists( exampleFn ) )
      std::filesystem::remove( exampleFn );

   // Write example payload to disk with unbuffered GAMS stream
   {
      StrmClazz fs { exampleFn, fmOpenWrite };
      fs.WriteByte( examplePayload );
   }

   REQUIRE( std::filesystem::exists( exampleFn ) );

   // Use C++ standard library file stream to slurp file and compare payload contents
   {
      std::ifstream ifs { exampleFn };
      char buf;
      ifs.read( &buf, 1 );
      REQUIRE_EQ( examplePayload, buf );
      REQUIRE_EQ( EOF, ifs.peek() );
   }

   std::filesystem::remove( exampleFn );
}

TEST_CASE( "Use unbuffered file stream to write a file" )
{
   testWritingOneByteIntoTextFile<TXFileStream>();
}

TEST_CASE( "Use buffered file stream to write a file" )
{
   testWritingOneByteIntoTextFile<TBufferedFileStream>();
}

TEST_CASE( "Use binary text file IO to write an uncompressed file" )
{
   const bool compression {}, password {};
   const auto examplePayload = 23;
   const std::string exampleFn { "testfile"s + ( compression ? ".gz"s : ""s ) };
   const std::string pw = password ? "mypass"s : ""s;

   if( std::filesystem::exists( exampleFn ) )
      std::filesystem::remove( exampleFn );

   {
      int errNr;
      std::string errMsg;
      TBinaryTextFileIO fs { exampleFn, "testCppMEX"s, pw, compression ? TFileSignature::fsign_gzip : TFileSignature::fsign_text, compression, errNr, errMsg };
      REQUIRE( errMsg.empty() );
      char buf { 23 };
      REQUIRE_EQ( 1u, fs.Write( &buf, 1 ) );
   };

   REQUIRE( std::filesystem::exists( exampleFn ) );

   {
      std::ifstream ifs { exampleFn, std::ios::binary };
      char buf {};
      ifs.read( &buf, 1 );
      REQUIRE_EQ( examplePayload, buf );
      REQUIRE_EQ( EOF, ifs.peek() );
   }

   std::filesystem::remove( exampleFn );
}

static void testWritingSingleLineModel( const bool useCompression, const bool usePassword )
{
   std::string miniModelText = "scalar x / 23 /;"s;

   int ErrNr;
   std::string ErrMsg;
   const std::string mypass = usePassword ? "strenggeheim"s : ""s, model_fn = "test.gms"s;

   if( std::filesystem::exists( model_fn ) )
      std::filesystem::remove( model_fn );

   {
      TBinaryTextFileIO Fout { model_fn, "SingleLineCompressDecompress"s, mypass, fsign_text, useCompression, ErrNr, ErrMsg };
      Fout.Write( miniModelText.data(), static_cast<int>(miniModelText.size()) );
   }

   ErrMsg.clear();

   {
      TBinaryTextFileIO Fin { model_fn, mypass, ErrNr, ErrMsg };
      std::array<char, 4096> readBuf {};
      auto len = Fin.Read( readBuf.data(), static_cast<uint32_t>(readBuf.size()) );
      readBuf[len] = '\0';
      std::string decompressedText( readBuf.data() );
      REQUIRE_EQ( miniModelText, decompressedText );
   }

   std::filesystem::remove( model_fn );
}

TEST_CASE( "Writing a single line model without compression" )
{
   testWritingSingleLineModel( false, false );
}

TEST_CASE( "Writing a single line model with compression" )
{
   testWritingSingleLineModel( true, false );
}

TEST_CASE( "Buffered scalar reads cross blocks and preserve byte order" )
{
   bool compressed = false, swapped = false;
   SUBCASE( "uncompressed native" ) {}
   SUBCASE( "compressed native" ) { compressed = true; }
   SUBCASE( "uncompressed swapped" ) { swapped = true; }
   SUBCASE( "compressed swapped" ) { compressed = swapped = true; }

   const std::string filename { "buffered-scalars.bin" };
   int64_t dataStart {};
   {
      TBufferedFileStream out { filename, fmCreate };
      const auto write = [&]( auto value ) {
         std::array<uint8_t, sizeof( value )> bytes;
         std::memcpy( bytes.data(), &value, bytes.size() );
         if( swapped ) std::reverse( bytes.begin(), bytes.end() );
         REQUIRE_EQ( out.Write( bytes.data(), bytes.size() ), bytes.size() );
      };
      // Explicit byte-order header lets the same payload exercise both
      // native and foreign-endian files on every host architecture.
      write( uint8_t { sizeof( uint16_t ) } );
      write( PAT_WORD );
      write( uint8_t { sizeof( int ) } );
      write( PAT_INTEGER );
      write( uint8_t { sizeof( double ) } );
      write( PAT_DOUBLE );
      REQUIRE( out.FlushBuffer() );
      dataStart = out.GetPosition();
      out.SetCompression( compressed );
      // Each tuple is 23 bytes. Over 32769 tuples every scalar alignment
      // crosses a 32 KiB boundary, including a byte at the next block start.
      for( int i {}; i <= BufferSize; ++i )
      {
         write( static_cast<uint8_t>( i ) );
         write( static_cast<uint16_t>( i * 3 ) );
         write( -i * 12345 );
         write( int64_t { -123456789012345LL } + i );
         write( -i - 0.25 );
      }
      REQUIRE( out.FlushBuffer() );
   }
   {
      TMiBufferedStream in { filename, fmOpenRead };
      REQUIRE_EQ( in.GoodByteOrder(), 0 );
      in.SetCompression( compressed );
      in.SetPosition( dataStart );
      for( int i {}; i <= BufferSize; ++i )
      {
         REQUIRE_EQ( in.ReadByte(), static_cast<uint8_t>( i ) );
         const auto block = in.GetCheckpointBlock();
         const auto offset = in.GetCheckpointOffset();
         const auto check = [&]() {
            REQUIRE_EQ( in.ReadWord(), static_cast<uint16_t>( i * 3 ) );
            REQUIRE_EQ( in.ReadInteger(), -i * 12345 );
            REQUIRE_EQ( in.ReadInt64(), int64_t { -123456789012345LL } + i );
            REQUIRE_EQ( in.ReadDouble(), -i - 0.25 );
         };
         check();
         // Resuming within a compressed block must not retain stale
         // buffered state. Also cover tuples that straddle the boundary.
         if( compressed && ( i % 997 == 0 || offset > BufferSize - 23 ) )
         {
            in.SetCheckpoint( block, offset );
            check();
         }
      }
      REQUIRE_EQ( in.GetLastIOResult(), 0 );
   }
   std::filesystem::remove( filename );
}

TEST_SUITE_END();

}
