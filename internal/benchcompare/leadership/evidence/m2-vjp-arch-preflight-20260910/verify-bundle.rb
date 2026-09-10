require 'digest'
require 'json'
require 'open3'

directory = '/private/tmp/goai-vjp-arch-publication-VYh1xG'
archive = File.join(directory, 'evidence.tar.zst')
manifest_bytes = File.binread(File.join(directory, 'manifest.json'))
manifest = JSON.parse(manifest_bytes)
artifacts = manifest.fetch('artifacts')
blobs = artifacts.group_by { |a| a.fetch('blob') }
sources = manifest.fetch('sources').merge(
  'root-symbolic' => '/private/tmp/goai-vjp-arch-symbolic-recovery-DJ1swR',
  'publication' => directory)
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

command = ['/opt/homebrew/bin/zstd', '-d', '-c', '--check', archive]
File.write(File.join(directory, 'verify-command.json'), JSON.pretty_generate({argv: command, cwd: Dir.pwd}) + "\n")
Open3.popen3(*command) do |input, stream, error, process|
  input.close
  loop do
    header = read_exact(stream, 512)
    if header == "\0" * 512
      abort 'bad tar terminator' unless read_exact(stream, 512) == "\0" * 512
      abort 'trailing archive data' unless stream.read.to_s.empty?
      break
    end
    checksum = header.byteslice(148, 8).to_i(8)
    blanked = header.dup
    blanked[148, 8] = ' ' * 8
    abort 'tar checksum mismatch' unless blanked.bytes.sum == checksum
    name = header.byteslice(0, 100).split("\0").first
    type = header.byteslice(156, 1)
    abort 'non-regular archive entry' unless ["0", "\0"].include?(type)
    abort 'duplicate archive entry' if seen[name]
    size = header.byteslice(124, 12).to_i(8)
    bytes = read_exact(stream, size)
    padding = read_exact(stream, (512 - size % 512) % 512)
    abort 'nonzero tar padding' unless padding.bytes.all?(&:zero?)
    if name == 'manifest.json'
      abort 'manifest mismatch' unless bytes == manifest_bytes
    else
      entries = blobs.fetch(name) { abort "unknown blob #{name}" }
      sha = Digest::SHA256.hexdigest(bytes)
      entries.each do |entry|
        abort 'descriptor mismatch' unless entry.fetch('sha256') == sha && entry.fetch('bytes') == size
        label = entry.fetch('name')
        namespace = sources.keys.sort_by { |key| -key.length }.find { |key| label.start_with?(key + '/') }
        abort 'unknown source namespace' unless namespace
        path = File.join(sources.fetch(namespace), label.delete_prefix(namespace + '/'))
        abort "original differs: #{label}" unless File.binread(path) == bytes
        decoded += size
      end
    end
    seen[name] = true
  end
  stderr = error.read
  result = process.value
  File.binwrite(File.join(directory, 'verify-zstd.stderr'), stderr)
  File.write(File.join(directory, 'verify-zstd.exit'), result.exitstatus.to_s + "\n")
  abort 'decompression failed' unless result.success?
end
abort 'missing archive entries' unless seen.keys.sort == (blobs.keys + ['manifest.json']).sort
puts JSON.pretty_generate({artifacts: artifacts.length, unique_blobs: blobs.length,
  decoded_artifact_bytes: decoded, every_hash_and_original_byte_verified: true,
  archive_bytes: File.size(archive), archive_sha256: Digest::SHA256.file(archive).hexdigest,
  manifest_sha256: Digest::SHA256.hexdigest(manifest_bytes)})
