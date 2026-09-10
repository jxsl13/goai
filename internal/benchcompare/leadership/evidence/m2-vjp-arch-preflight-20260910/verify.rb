# Stream-verify the published archive without extracting its payload.
require 'digest'
require 'json'
require 'open3'

directory = File.expand_path(__dir__)
archive = File.join(directory, 'evidence.tar.zst')
manifest_bytes = File.binread(File.join(directory, 'manifest.json'))
abort 'archive hash mismatch' unless Digest::SHA256.file(archive).hexdigest == '8405e6319df2e4c1917b077b743b4cf1af9bff1c109dc8289ca2f37c0088f838'
abort 'manifest hash mismatch' unless Digest::SHA256.hexdigest(manifest_bytes) == '8b22a13fd877b03110e302ddd6340c5df9e38f459fa0cf04a2b1ad404afc2617'
manifest = JSON.parse(manifest_bytes)
abort 'unsupported format' unless manifest.fetch('format') == 'content-addressed-tar-v1'
artifacts = manifest.fetch('artifacts')
abort 'duplicate logical name' unless artifacts.map { |a| a.fetch('name') }.uniq.length == artifacts.length
blobs = artifacts.group_by { |a| a.fetch('blob') }
seen = {}
decoded = 0

def read_exact(stream, size)
  result = ''.b
  while result.bytesize < size
    part = stream.read(size - result.bytesize)
    abort 'truncated tar' unless part && !part.empty?
    result << part
  end
  result
end

Open3.popen3('zstd', '-d', '-c', '--check', archive) do |input, stream, error, process|
  input.close
  errors = Thread.new { error.read }
  loop do
    header = read_exact(stream, 512)
    if header == "\0" * 512
      abort 'bad tar terminator' unless read_exact(stream, 512) == "\0" * 512
      abort 'trailing archive data' unless stream.read.to_s.empty?
      break
    end
    blanked = header.dup
    blanked[148, 8] = ' ' * 8
    abort 'tar checksum mismatch' unless blanked.bytes.sum == header.byteslice(148, 8).to_i(8)
    name = header.byteslice(0, 100).split("\0").first
    abort 'non-regular entry' unless ["0", "\0"].include?(header.byteslice(156, 1))
    abort 'duplicate entry' if seen[name]
    size = header.byteslice(124, 12).to_i(8)
    bytes = read_exact(stream, size)
    abort 'nonzero padding' unless read_exact(stream, (512 - size % 512) % 512).bytes.all?(&:zero?)
    if name == 'manifest.json'
      abort 'manifest mismatch' unless bytes == manifest_bytes
    else
      entries = blobs.fetch(name) { abort "unknown blob #{name}" }
      sha = Digest::SHA256.hexdigest(bytes)
      abort 'incorrect blob name' unless name == 'blobs/' + sha
      entries.each do |entry|
        abort 'descriptor mismatch' unless entry.fetch('sha256') == sha && entry.fetch('bytes') == size
        decoded += size
      end
    end
    seen[name] = true
  end
  result = process.value
  stderr = errors.value
  abort "decompression failed: #{stderr}" unless result.success?
end
abort 'missing entries' unless seen.keys.sort == (blobs.keys + ['manifest.json']).sort
puts JSON.pretty_generate({artifacts: artifacts.length, unique_blobs: blobs.length,
  decoded_artifact_bytes: decoded, all_published_hashes_verified: true})
